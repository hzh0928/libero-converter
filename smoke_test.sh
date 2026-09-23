#!/usr/bin/env bash
# LIBERO -> FastWAM conversion smoke test.
#
# Downloads exactly ONE raw task HDF5 (libero_spatial, first task), replays
# ONE demo for 20 frames, converts to LeRobot v2.1 + HDF5 sidecar, and runs
# --stage validate. On ANY failure, safely removes only the paths this
# script itself created (each tracked directory is marked with a unique
# signature file before deletion is even considered, so pre-existing/other
# users' data is never touched).
#
# Usage:
#   bash smoke_test.sh
#
set -uo pipefail

WORKDIR="/root/workspace/libero_converter_smoketest"
DATA_RAW_DIR="/root/dataset/libero_raw_smoketest"
SMOKE_OUT_DIR="/root/dataset/libero_smoke_test"
LIBERO_SRC_DIR="/root/LIBERO_smoketest_src"
SUITE="libero_spatial"
MARKER=".smoke_test_marker_8f2c91"
LOG_FILE="/root/workspace/smoke_test_$(date +%Y%m%d_%H%M%S).log"

mkdir -p /root/workspace
: > "$LOG_FILE"

CREATED_DIRS=()      # directories we created; only removed if marker file is present inside
DOWNLOADED_FILES=()  # individual files we downloaded; removed by exact path only

log() { echo "[$(date +%T)] $*" | tee -a "$LOG_FILE"; }

# ---- Environment bootstrap: conda + outbound proxy (non-interactive shell, .bashrc not sourced) ----
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
  log "Sourced conda.sh from /opt/conda"
fi
if [ -f /root/clashctl/scripts/cmd/clashctl.sh ]; then
  set +u
  # shellcheck disable=SC1091
  source /root/clashctl/scripts/cmd/clashctl.sh
  clashon >>"$LOG_FILE" 2>&1 || true
  set -u
  export http_proxy="http://127.0.0.1:7890"
  export https_proxy="http://127.0.0.1:7890"
  export HTTP_PROXY="$http_proxy"
  export HTTPS_PROXY="$https_proxy"
  export no_proxy="localhost,127.0.0.1,::1"
  export NO_PROXY="$no_proxy"
  log "Proxy environment exported (http://127.0.0.1:7890)"
fi

cleanup() {
  log "---- Safety cleanup starting (whitelist + marker verification) ----"
  local p
  for p in "${CREATED_DIRS[@]:-}"; do
    [ -z "$p" ] && continue
    if [ -d "$p" ] && [ -f "$p/$MARKER" ]; then
      log "Removing directory (marker verified): $p"
      rm -rf "$p"
    elif [ -d "$p" ]; then
      log "SKIP $p: marker file missing, refusing to delete for safety"
    fi
  done
  local f
  for f in "${DOWNLOADED_FILES[@]:-}"; do
    [ -z "$f" ] && continue
    if [ -f "$f" ]; then
      log "Removing downloaded test file: $f"
      rm -f "$f"
    fi
  done
  # remove now-empty raw-data directories we may have created, never non-empty ones
  if [ -d "$DATA_RAW_DIR/$SUITE" ] && [ -z "$(ls -A "$DATA_RAW_DIR/$SUITE" 2>/dev/null)" ]; then
    rmdir "$DATA_RAW_DIR/$SUITE" 2>/dev/null || true
  fi
  if [ -d "$DATA_RAW_DIR" ] && [ -z "$(ls -A "$DATA_RAW_DIR" 2>/dev/null)" ]; then
    rmdir "$DATA_RAW_DIR" 2>/dev/null || true
  fi
  log "---- Cleanup done ----"
}

fail_and_cleanup() {
  log "!!! FAILURE: $1"
  cleanup
  log "Smoke test FAILED and was cleaned up. Full log: $LOG_FILE"
  exit 1
}

# ---- Safety: never touch pre-existing paths ----
for existing in "$WORKDIR" "$SMOKE_OUT_DIR" "$DATA_RAW_DIR" "$LIBERO_SRC_DIR"; do
  if [ -e "$existing" ]; then
    echo "ABORT: $existing already exists. Refusing to run to avoid clobbering existing data." >&2
    echo "Remove/rename it first, or edit WORKDIR/SMOKE_OUT_DIR in this script." >&2
    exit 2
  fi
done

mkdir -p "$WORKDIR"
touch "$WORKDIR/$MARKER"
CREATED_DIRS+=("$WORKDIR")
log "Created workdir: $WORKDIR"

# ---- Write the conversion script + requirements into WORKDIR ----
cat > "$WORKDIR/convert_libero_fastwam.py" <<'PYEOF'
#!/usr/bin/env python3
"""Replay LIBERO demonstrations and export FastWAM-compatible LeRobot v2.1.

The normal LeRobot fields match yuanty/LIBERO-fastwam. Numeric 3D annotations
(depth, masks, camera matrices and raw instance segmentation) are kept in
HDF5 sidecars under each output suite's ``annotations_3d/tasks`` directory.

Run with MUJOCO_GL=egl on a headless NVIDIA machine.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# Render images in OpenCV row order (top-left origin). This is critical for
# depth/unprojection consistency (see LIBERO issue #101): robosuite's default
# "opengl" convention returns vertically flipped observations.
import robosuite.macros as macros

macros.IMAGE_CONVENTION = "opencv"

SCHEMA_VERSION = 1
FPS = 20
CAMERAS = ("agentview", "robot0_eye_in_hand")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
ROBOT_INSTANCE_NAMES = {"Panda0", "PandaGripper0", "RethinkMount0"}


def log(message: str) -> None:
    print(message, flush=True)


def natural_demo_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[1]), name
    except (IndexError, ValueError):
        return sys.maxsize, name


def is_noop(action: np.ndarray, previous_kept: np.ndarray | None, threshold: float) -> bool:
    if previous_kept is None:
        return False
    arm_is_still = np.linalg.norm(action[:-1]) < threshold
    return bool(arm_is_still and np.isclose(action[-1], previous_kept[-1]))


def keep_indices(actions: np.ndarray, remove_noops: bool, threshold: float) -> np.ndarray:
    if not remove_noops:
        return np.arange(len(actions), dtype=np.int64)
    kept: list[int] = []
    previous: np.ndarray | None = None
    for index, action in enumerate(actions):
        if not is_noop(action, previous, threshold):
            kept.append(index)
            previous = action
    return np.asarray(kept, dtype=np.int64)


def fastwam_action(action: np.ndarray) -> np.ndarray:
    """Map LIBERO gripper (-1 open, +1 close) to FastWAM (1 open, 0 close)."""
    result = np.asarray(action, dtype=np.float32).copy()
    result[-1] = 1.0 - np.clip(result[-1], 0.0, 1.0)
    return result


def rotate_calibration_180(
    intrinsic: np.ndarray, extrinsic_camera_to_world: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Calibration for a 180-degree image rotation (u,v) -> (W-1-u, H-1-v)."""
    intrinsic = np.asarray(intrinsic, dtype=np.float64).copy()
    intrinsic[0, 2] = (width - 1) - intrinsic[0, 2]
    intrinsic[1, 2] = (height - 1) - intrinsic[1, 2]
    camera_rotation = np.eye(4, dtype=np.float64)
    camera_rotation[0, 0] = -1.0
    camera_rotation[1, 1] = -1.0
    return intrinsic, np.asarray(extrinsic_camera_to_world) @ camera_rotation


def transform_image(array: np.ndarray, rotate_180: bool) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if rotate_180:
        # With IMAGE_CONVENTION="opencv" the observation is already in OpenCV
        # orientation, so this is a true 180-degree rotation.
        array = array[::-1, ::-1]
    return np.ascontiguousarray(array)


def parse_problem_info(data_group: h5py.Group, fallback: str) -> str:
    raw = data_group.attrs.get("problem_info")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if raw:
        try:
            return str(json.loads(raw)["language_instruction"])
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return fallback.replace("_", " ")


def instance_mapping(env: Any) -> tuple[dict[int, str], set[int], set[int]]:
    names = list(env.env.model.instances_to_ids.keys())
    id_to_name = {index + 1: str(name) for index, name in enumerate(names)}
    robot_ids = {
        instance_id
        for instance_id, name in id_to_name.items()
        if name in ROBOT_INSTANCE_NAMES
    }
    interest_ids = {
        instance_id
        for name in env.obj_of_interest
        for instance_id, instance_name in id_to_name.items()
        if instance_name == name
    }
    return id_to_name, robot_ids, interest_ids


def output_name(suite_name: str, keep_noops: bool) -> str:
    suffix = "lerobot" if keep_noops else "no_noops_lerobot"
    return f"{suite_name}_{suffix}"


_ASSET_ROOTS: list[str] | None = None


def local_asset_roots() -> list[str]:
    """Return candidate local roots for MuJoCo assets shipped with LIBERO/robosuite."""
    global _ASSET_ROOTS
    if _ASSET_ROOTS is None:
        roots: list[str] = []
        try:
            from libero.libero import get_libero_path

            roots.append(str(Path(get_libero_path("assets"))))
        except Exception:  # pragma: no cover - depends on install layout
            pass
        try:
            import robosuite

            roots.append(str(Path(robosuite.__file__).parent / "models" / "assets"))
        except Exception:  # pragma: no cover
            pass
        _ASSET_ROOTS = roots
    return _ASSET_ROOTS


def remap_asset_paths(xml: str) -> str:
    """Rewrite absolute asset paths baked into the recorded demo files.

    LIBERO demo HDF5s store a MuJoCo model XML that still references absolute
    paths on the machine that recorded the demonstrations (for example
    ``/Users/yifengz/workspace/libero-dev/chiliocosm/assets/...``). Those files
    do not exist on other machines, so MuJoCo fails while loading meshes. Any
    missing absolute path is remapped onto the local LIBERO / robosuite asset
    roots by matching its ``/assets/`` suffix.
    """
    import re

    roots = local_asset_roots()
    if not roots:
        return xml

    def replace(match):
        path = match.group(1)
        if not path.startswith("/") or os.path.exists(path):
            return match.group(0)
        marker = "/assets/"
        index = path.find(marker)
        if index < 0:
            return match.group(0)
        relative = path[index + len(marker):]
        for root in roots:
            candidate = os.path.join(root, relative)
            if os.path.exists(candidate):
                return 'file="%s"' % candidate
        return match.group(0)

    return re.sub(r'file="([^"]+)"', replace, xml)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def render_config(args: argparse.Namespace, source_file: Path, bddl_file: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "fps": FPS,
        "cameras": list(CAMERAS),
        "resolution": args.resolution,
        "remove_noops": not args.keep_noops,
        "noop_threshold": args.noop_threshold,
        "rotate_180": not args.no_rotate,
        "replay_mode": args.replay_mode,
        "settle_steps": args.settle_steps,
        "max_state_error": args.max_state_error,
        "max_demos": args.max_demos,
        "max_frames": args.max_frames,
        "source_sha256": sha256_file(source_file),
        "bddl_sha256": sha256_file(bddl_file),
        "mujoco_version": package_version("mujoco"),
        "robosuite_version": package_version("robosuite"),
        "libero_version": package_version("libero"),
    }


def create_episode_datasets(group: h5py.Group, length: int, resolution: int, state_dim: int) -> dict[str, Any]:
    image_chunks = (1, 1, resolution, resolution)
    rgb_chunks = (1, 1, resolution, resolution, 3)
    return {
        "rgb": group.create_dataset(
            "rgb", (length, 2, resolution, resolution, 3), np.uint8,
            chunks=rgb_chunks, compression="gzip", compression_opts=1,
        ),
        "depth": group.create_dataset(
            "depth", (length, 2, resolution, resolution), np.float16,
            chunks=image_chunks, compression="gzip", compression_opts=4,
        ),
        "robot_mask": group.create_dataset(
            "robot_mask", (length, 2, resolution, resolution), np.uint8,
            chunks=image_chunks, compression="gzip", compression_opts=4,
        ),
        "object_mask": group.create_dataset(
            "object_mask", (length, 2, resolution, resolution), np.uint8,
            chunks=image_chunks, compression="gzip", compression_opts=4,
        ),
        "object_of_interest_mask": group.create_dataset(
            "object_of_interest_mask", (length, 2, resolution, resolution), np.uint8,
            chunks=image_chunks, compression="gzip", compression_opts=4,
        ),
        "instance_mask": group.create_dataset(
            "instance_mask", (length, 2, resolution, resolution), np.uint16,
            chunks=image_chunks, compression="gzip", compression_opts=4,
        ),
        "extrinsics": group.create_dataset("extrinsics", (length, 2, 4, 4), np.float32),
        "source_state": group.create_dataset("source_state", (length, state_dim), np.float64),
        "replayed_sim_state": group.create_dataset("replayed_sim_state", (length, state_dim), np.float64),
        "state": group.create_dataset("state", (length, 8), np.float32),
        "ee_state": group.create_dataset("ee_state", (length, 6), np.float32),
        "joint_state": group.create_dataset("joint_state", (length, 7), np.float32),
        "gripper_state": group.create_dataset("gripper_state", (length, 2), np.float32),
        "action": group.create_dataset("action", (length, 7), np.float32),
        "source_action": group.create_dataset("source_action", (length, 7), np.float32),
        "source_frame_index": group.create_dataset("source_frame_index", (length,), np.int64),
    }


def replay_task(
    suite_name: str,
    task_id: int,
    task: Any,
    source_file: Path,
    target_file: Path,
    resolution: int,
    remove_noops: bool,
    noop_threshold: float,
    rotate_180: bool,
    replay_mode: str,
    settle_steps: int,
    max_state_error_limit: float,
    max_demos: int | None,
    max_frames: int | None,
    conversion_config: dict[str, Any],
) -> None:
    from libero.libero import get_libero_path
    from libero.libero.envs import SegmentationRenderEnv
    from libero.libero.utils.utils import postprocess_model_xml
    from robosuite.utils import camera_utils
    from robosuite.utils.transform_utils import quat2axisangle

    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = SegmentationRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_names=list(CAMERAS),
        camera_heights=resolution,
        camera_widths=resolution,
        camera_depths=True,
        camera_segmentations="instance",
        control_freq=FPS,
    )
    env.seed(0)
    env.reset()
    id_to_name, robot_ids, interest_ids = instance_mapping(env)
    all_object_ids = set(id_to_name).difference(robot_ids)

    raw_intrinsics = np.stack(
        [camera_utils.get_camera_intrinsic_matrix(env.sim, name, resolution, resolution) for name in CAMERAS]
    ).astype(np.float32)
    saved_intrinsics = raw_intrinsics.copy()
    if rotate_180:
        for camera_index in range(len(CAMERAS)):
            saved_intrinsics[camera_index], _ = rotate_calibration_180(
                raw_intrinsics[camera_index], np.eye(4), resolution, resolution
            )

    target_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = target_file.with_suffix(target_file.suffix + ".tmp")
    temporary_file.unlink(missing_ok=True)

    try:
        with h5py.File(source_file, "r") as source, h5py.File(temporary_file, "w") as target:
            source_data = source["data"]
            target.attrs.update(
                suite=suite_name,
                task_id=task_id,
                task_name=task.name,
                language_instruction=parse_problem_info(source_data, task.name),
                source_file=str(source_file),
                fps=FPS,
                resolution=resolution,
                camera_names=json.dumps(CAMERAS),
                depth_unit="meter",
                depth_definition="metric z-depth along OpenCV camera +Z",
                image_convention="opencv",
                extrinsics_convention="camera_to_world OpenCV (+X right, +Y down, +Z forward)",
                image_rotation_degrees=180 if rotate_180 else 0,
                instance_id_to_name=json.dumps(id_to_name, ensure_ascii=False),
                robot_instance_ids=json.dumps(sorted(robot_ids)),
                object_instance_ids=json.dumps(sorted(all_object_ids)),
                object_of_interest_instance_ids=json.dumps(sorted(interest_ids)),
                object_of_interest_names=json.dumps(list(env.obj_of_interest), ensure_ascii=False),
                conversion_config=json.dumps(conversion_config, sort_keys=True),
            )
            target.create_dataset("intrinsics", data=saved_intrinsics)
            target.create_dataset("intrinsics_before_image_rotation", data=raw_intrinsics)

            demos = sorted(source_data.keys(), key=natural_demo_key)
            if max_demos is not None:
                demos = demos[:max_demos]

            output_demo_index = 0
            for source_demo_name in demos:
                source_demo = source_data[source_demo_name]
                actions = np.asarray(source_demo["actions"])
                states = np.asarray(source_demo["states"])
                if actions.ndim != 2 or actions.shape[1] != 7:
                    raise ValueError(f"Expected (T, 7) actions in {source_demo_name}, got {actions.shape}")
                if states.ndim != 2:
                    raise ValueError(f"Expected (T, D) states in {source_demo_name}, got {states.shape}")
                if len(states) < len(actions):
                    raise ValueError(
                        f"Fewer states than actions in {source_demo_name}: "
                        f"states={len(states)}, actions={len(actions)}"
                    )
                count = len(actions)
                if count == 0:
                    log(f"  [skip] {source_demo_name}: empty action sequence")
                    continue
                sample_states = states[:count]
                terminal_state = states[count] if len(states) > count else None
                indices = keep_indices(actions, remove_noops, noop_threshold)
                if max_frames is not None:
                    indices = indices[:max_frames]
                if not len(indices):
                    log(f"  [skip] {source_demo_name}: no frames")
                    continue

                env.reset()
                model_xml = source_demo.attrs.get("model_file")
                if isinstance(model_xml, bytes):
                    model_xml = model_xml.decode("utf-8")
                if model_xml:
                    env.reset_from_xml_string(
                        remap_asset_paths(postprocess_model_xml(model_xml, {}))
                    )
                    env.sim.reset()
                obs = env.set_init_state(sample_states[0])
                for _ in range(settle_steps):
                    obs, _, _, _ = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

                demo_id_to_name, demo_robot_ids, demo_interest_ids = instance_mapping(env)
                demo_object_ids = set(demo_id_to_name).difference(demo_robot_ids)
                demo_intrinsics = np.stack(
                    [
                        camera_utils.get_camera_intrinsic_matrix(env.sim, name, resolution, resolution)
                        for name in CAMERAS
                    ]
                ).astype(np.float32)
                if not np.allclose(demo_intrinsics, raw_intrinsics, rtol=1e-5, atol=1e-5):
                    raise RuntimeError(f"Camera intrinsics changed in {source_demo_name}")

                demo = target.create_group(f"demo_{output_demo_index}")
                demo.attrs["source_demo"] = source_demo_name
                demo.attrs["num_samples"] = len(indices)
                demo.attrs["replay_mode"] = replay_mode
                demo.attrs["instance_id_to_name"] = json.dumps(demo_id_to_name, ensure_ascii=False)
                demo.attrs["robot_instance_ids"] = json.dumps(sorted(demo_robot_ids))
                demo.attrs["object_instance_ids"] = json.dumps(sorted(demo_object_ids))
                demo.attrs["object_of_interest_instance_ids"] = json.dumps(sorted(demo_interest_ids))
                datasets = create_episode_datasets(demo, len(indices), resolution, sample_states.shape[1])
                selected = {int(source_index): output_index for output_index, source_index in enumerate(indices)}
                replay_complete = max_frames is None or len(indices) == len(
                    keep_indices(actions, remove_noops, noop_threshold)
                )
                replay_success = False
                max_state_error = 0.0

                def write_frame(output_index: int, source_index: int, observation: dict[str, Any]) -> None:
                    nonlocal max_state_error
                    replayed_sim_state = env.sim.get_state().flatten()
                    state_error = float(np.linalg.norm(sample_states[source_index] - replayed_sim_state))
                    max_state_error = max(max_state_error, state_error)
                    datasets["source_state"][output_index] = sample_states[source_index]
                    datasets["replayed_sim_state"][output_index] = replayed_sim_state
                    datasets["source_action"][output_index] = actions[source_index]
                    datasets["action"][output_index] = fastwam_action(actions[source_index])
                    datasets["source_frame_index"][output_index] = source_index

                    ee_state = np.concatenate(
                        [
                            observation["robot0_eef_pos"],
                            quat2axisangle(observation["robot0_eef_quat"]),
                        ]
                    ).astype(np.float32)
                    gripper_state = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32)
                    joint_state = np.asarray(observation["robot0_joint_pos"], dtype=np.float32)
                    datasets["ee_state"][output_index] = ee_state
                    datasets["gripper_state"][output_index] = gripper_state
                    datasets["joint_state"][output_index] = joint_state
                    datasets["state"][output_index] = np.concatenate([ee_state, gripper_state])

                    for camera_index, camera_name in enumerate(CAMERAS):
                        rgb = transform_image(
                            observation[f"{camera_name}_image"], rotate_180
                        ).astype(np.uint8)
                        normalized_depth = np.asarray(observation[f"{camera_name}_depth"])
                        metric_depth = camera_utils.get_real_depth_map(env.sim, normalized_depth)
                        metric_depth = transform_image(metric_depth, rotate_180).astype(np.float16)
                        segmentation = transform_image(
                            observation[f"{camera_name}_segmentation_instance"], rotate_180
                        ).astype(np.uint16)
                        unknown_ids = set(np.unique(segmentation)).difference({0}, demo_id_to_name)
                        if unknown_ids:
                            raise RuntimeError(
                                f"Unknown segmentation IDs {sorted(unknown_ids)} in {source_demo_name}"
                            )

                        extrinsic = camera_utils.get_camera_extrinsic_matrix(env.sim, camera_name)
                        if rotate_180:
                            _, extrinsic = rotate_calibration_180(
                                raw_intrinsics[camera_index], extrinsic, resolution, resolution
                            )

                        datasets["rgb"][output_index, camera_index] = rgb
                        datasets["depth"][output_index, camera_index] = metric_depth
                        datasets["instance_mask"][output_index, camera_index] = segmentation
                        datasets["robot_mask"][output_index, camera_index] = np.isin(
                            segmentation, list(demo_robot_ids)
                        ).astype(np.uint8)
                        datasets["object_mask"][output_index, camera_index] = np.isin(
                            segmentation, list(demo_object_ids)
                        ).astype(np.uint8)
                        datasets["object_of_interest_mask"][output_index, camera_index] = np.isin(
                            segmentation, list(demo_interest_ids)
                        ).astype(np.uint8)
                        datasets["extrinsics"][output_index, camera_index] = extrinsic

                if replay_mode == "action":
                    replay_limit = count if replay_complete else int(indices[-1]) + 1
                    for source_index in range(replay_limit):
                        output_index = selected.get(source_index)
                        if output_index is not None:
                            write_frame(output_index, source_index, obs)
                        obs, _, _, _ = env.step(actions[source_index].tolist())
                        replay_success = replay_success or bool(env.check_success())
                else:
                    for output_index, source_index in enumerate(indices):
                        obs = env.set_init_state(sample_states[source_index])
                        write_frame(output_index, int(source_index), obs)
                        replay_success = replay_success or bool(env.check_success())
                    if replay_complete:
                        if terminal_state is not None:
                            env.set_init_state(terminal_state)
                        else:
                            env.set_init_state(sample_states[-1])
                            env.step(actions[-1].tolist())
                        replay_success = replay_success or bool(env.check_success())

                demo.attrs["replay_complete"] = replay_complete
                demo.attrs["replay_success"] = replay_success
                demo.attrs["max_state_error"] = max_state_error
                if replay_mode == "action" and settle_steps == 0 and max_state_error > max_state_error_limit:
                    raise RuntimeError(
                        f"Action replay diverged in {source_demo_name}: "
                        f"max state error {max_state_error:.6f} > {max_state_error_limit:.6f}. "
                        "Use the matching LIBERO/MuJoCo versions or --replay-mode state."
                    )
                log(
                    f"  [rendered] {source_demo_name}: {len(indices)} frames, "
                    f"success={replay_success}, max_state_error={max_state_error:.6f}"
                )
                output_demo_index += 1

            target.attrs["num_demos"] = output_demo_index
            if output_demo_index == 0:
                raise RuntimeError(f"No demonstrations exported from {source_file}")
        os.replace(temporary_file, target_file)
    finally:
        temporary_file.unlink(missing_ok=True)
        env.close()


def replay_suites(args: argparse.Namespace) -> None:
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    for suite_name in args.suites:
        suite = benchmark_dict[suite_name]()
        source_dir = args.raw_dir / suite_name
        stage_dir = args.output_dir / ".staging_3d" / suite_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        task_ids = list(range(suite.n_tasks))
        if args.max_tasks is not None:
            task_ids = task_ids[: args.max_tasks]

        log(f"[suite] {suite_name}: {len(task_ids)} tasks")
        for task_id in task_ids:
            task = suite.get_task(task_id)
            source_file = source_dir / f"{task.name}_demo.hdf5"
            target_file = stage_dir / f"{task.name}.hdf5"
            if not source_file.exists():
                raise FileNotFoundError(f"Missing source task: {source_file}")
            bddl_file = Path(suite.get_task_bddl_file_path(task_id))
            config = render_config(args, source_file, bddl_file)
            if target_file.exists() and not args.overwrite_render:
                with h5py.File(target_file, "r") as rendered:
                    existing_config = json.loads(rendered.attrs.get("conversion_config", "{}"))
                if existing_config != config:
                    raise RuntimeError(
                        f"Existing staging file has different settings: {target_file}. "
                        "Use --overwrite-render or a different --output-dir."
                    )
                log(f"[skip] rendered task exists with matching settings: {target_file}")
                continue
            log(f"[render] {suite_name}/{task.name}")
            replay_task(
                suite_name=suite_name,
                task_id=task_id,
                task=task,
                source_file=source_file,
                target_file=target_file,
                resolution=args.resolution,
                remove_noops=not args.keep_noops,
                noop_threshold=args.noop_threshold,
                rotate_180=not args.no_rotate,
                replay_mode=args.replay_mode,
                settle_steps=args.settle_steps,
                max_state_error_limit=args.max_state_error,
                max_demos=args.max_demos,
                max_frames=args.max_frames,
                conversion_config=config,
            )


def lerobot_features(resolution: int) -> dict[str, dict[str, Any]]:
    motor_names = ["x", "y", "z", "roll", "pitch", "yaw"]
    return {
        "observation.images.image": {
            "dtype": "video", "shape": (3, resolution, resolution),
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist_image": {
            "dtype": "video", "shape": (3, resolution, resolution),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32", "shape": (8,),
            "names": {"motors": motor_names + ["gripper", "gripper"]},
        },
        "observation.states.ee_state": {
            "dtype": "float32", "shape": (6,), "names": {"motors": motor_names},
        },
        "observation.states.joint_state": {
            "dtype": "float32", "shape": (7,),
            "names": {"motors": [f"joint_{i}" for i in range(7)]},
        },
        "observation.states.gripper_state": {
            "dtype": "float32", "shape": (2,),
            "names": {"motors": ["gripper", "gripper"]},
        },
        "action": {
            "dtype": "float32", "shape": (7,),
            "names": {"motors": motor_names + ["gripper"]},
        },
    }


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def commit_directory(temporary_root: Path, output_root: Path) -> None:
    backup_root = output_root.with_name(f".{output_root.name}.backup")
    if backup_root.exists() and not output_root.exists():
        os.replace(backup_root, output_root)
    elif backup_root.exists():
        shutil.rmtree(backup_root)
    if output_root.exists():
        os.replace(output_root, backup_root)
    try:
        os.replace(temporary_root, output_root)
    except Exception:
        if backup_root.exists() and not output_root.exists():
            os.replace(backup_root, output_root)
        raise
    if backup_root.exists():
        shutil.rmtree(backup_root)


def build_lerobot_suite(args: argparse.Namespace, suite_name: str) -> None:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from libero.libero import benchmark
    except ImportError as error:
        raise RuntimeError("LeRobot v0.3.3 and LIBERO are required for --stage lerobot/all") from error

    stage_dir = args.output_dir / ".staging_3d" / suite_name
    suite = benchmark.get_benchmark_dict()[suite_name]()
    tasks = [suite.get_task(task_id) for task_id in range(suite.n_tasks)]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]
    task_files = [stage_dir / f"{task.name}.hdf5" for task in tasks]
    missing_tasks = [str(task_file) for task_file in task_files if not task_file.exists()]
    if missing_tasks:
        raise FileNotFoundError("Missing rendered task files:\n- " + "\n- ".join(missing_tasks))
    task_configs: list[dict[str, Any]] = []
    semantic_config: dict[str, Any] | None = None
    source_specific_keys = {"source_sha256", "bddl_sha256"}
    expected_semantic_config = {
        "schema_version": SCHEMA_VERSION,
        "fps": FPS,
        "cameras": list(CAMERAS),
        "resolution": args.resolution,
        "remove_noops": not args.keep_noops,
        "noop_threshold": args.noop_threshold,
        "rotate_180": not args.no_rotate,
        "replay_mode": args.replay_mode,
        "settle_steps": args.settle_steps,
        "max_state_error": args.max_state_error,
        "max_demos": args.max_demos,
        "max_frames": args.max_frames,
        "mujoco_version": package_version("mujoco"),
        "robosuite_version": package_version("robosuite"),
        "libero_version": package_version("libero"),
    }
    for task_file in task_files:
        with h5py.File(task_file, "r") as rendered:
            config = json.loads(rendered.attrs.get("conversion_config", "{}"))
            if int(rendered.attrs.get("resolution", -1)) != args.resolution:
                raise RuntimeError(f"Resolution mismatch in {task_file}")
            current_semantic = {key: value for key, value in config.items() if key not in source_specific_keys}
            if current_semantic != expected_semantic_config:
                raise RuntimeError(
                    f"Staging settings do not match this command in {task_file}. "
                    "Use matching options or rerun render with --overwrite-render."
                )
            if semantic_config is None:
                semantic_config = current_semantic
            elif current_semantic != semantic_config:
                raise RuntimeError(f"Mixed staging settings detected in {task_file}")
            task_configs.append({"file": task_file.name, "config": config})

    dataset_name = output_name(suite_name, args.keep_noops)
    output_root = args.output_dir / dataset_name
    temporary_root = args.output_dir / f".{dataset_name}.tmp"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "suite": suite_name,
        "keep_failed_replays": args.keep_failed_replays,
        "features": lerobot_features(args.resolution),
        "tasks": task_configs,
    }
    manifest_json = json.dumps(manifest, sort_keys=True)
    if output_root.exists() and not args.overwrite_lerobot:
        manifest_path = output_root / "meta" / "conversion_manifest.json"
        existing_manifest = manifest_path.read_text(encoding="utf-8") if manifest_path.exists() else ""
        if existing_manifest != manifest_json:
            raise RuntimeError(
                f"Existing output has different settings: {output_root}. "
                "Use --overwrite-lerobot or another --output-dir."
            )
        log(f"[skip] LeRobot suite exists with matching settings: {output_root}")
        return
    if temporary_root.exists():
        shutil.rmtree(temporary_root)

    dataset = LeRobotDataset.create(
        repo_id=f"local/{dataset_name}",
        root=temporary_root,
        fps=FPS,
        robot_type="franka",
        features=lerobot_features(args.resolution),
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=args.image_writer_threads,
        batch_encoding_size=1,
    )

    annotation_rows: list[dict[str, Any]] = []
    try:
        for task_file in task_files:
            with h5py.File(task_file, "r") as source:
                task = str(source.attrs["language_instruction"])
                demos = sorted((key for key in source if key.startswith("demo_")), key=natural_demo_key)
                for demo_name in demos:
                    demo = source[demo_name]
                    replay_complete = bool(demo.attrs.get("replay_complete", True))
                    replay_success = bool(demo.attrs.get("replay_success", True))
                    if not args.keep_failed_replays and replay_complete and not replay_success:
                        log(f"  [skip failed] {task_file.stem}/{demo_name}")
                        continue
                    episode_index = dataset.meta.total_episodes
                    length = int(demo.attrs["num_samples"])
                    for frame_index in range(length):
                        frame = {
                            "observation.images.image": np.asarray(demo["rgb"][frame_index, 0]),
                            "observation.images.wrist_image": np.asarray(demo["rgb"][frame_index, 1]),
                            "observation.state": np.asarray(demo["state"][frame_index], dtype=np.float32),
                            "observation.states.ee_state": np.asarray(
                                demo["ee_state"][frame_index], dtype=np.float32
                            ),
                            "observation.states.joint_state": np.asarray(
                                demo["joint_state"][frame_index], dtype=np.float32
                            ),
                            "observation.states.gripper_state": np.asarray(
                                demo["gripper_state"][frame_index], dtype=np.float32
                            ),
                            "action": np.asarray(demo["action"][frame_index], dtype=np.float32),
                        }
                        dataset.add_frame(frame=frame, task=task)
                    dataset.save_episode()
                    annotation_rows.append(
                        {
                            "episode_index": episode_index,
                            "file": f"annotations_3d/tasks/{task_file.name}",
                            "group": demo_name,
                            "source_demo": str(demo.attrs["source_demo"]),
                            "length": length,
                            "replay_complete": replay_complete,
                            "replay_success": replay_success,
                            "max_state_error": float(demo.attrs.get("max_state_error", 0.0)),
                        }
                    )
                    log(f"  [LeRobot] episode {episode_index}: {task_file.stem}/{demo_name} ({length})")
    finally:
        dataset.stop_image_writer()

    if not annotation_rows:
        raise RuntimeError(
            f"No successful episodes found for {suite_name}. "
            "Inspect replay_success or rerun with --keep-failed-replays."
        )

    annotation_dir = temporary_root / "annotations_3d" / "tasks"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    for task_file in task_files:
        link_or_copy(task_file, annotation_dir / task_file.name)
    metadata_file = temporary_root / "meta" / "annotations_3d.jsonl"
    with metadata_file.open("w", encoding="utf-8") as stream:
        for row in annotation_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (temporary_root / "meta" / "conversion_manifest.json").write_text(
        manifest_json, encoding="utf-8"
    )

    readme = temporary_root / "annotations_3d" / "README.txt"
    readme.write_text(
        "Each row in meta/annotations_3d.jsonl maps a LeRobot episode to an HDF5 group.\n"
        "HDF5 arrays: rgb, depth(m), robot_mask, object_mask, object_of_interest_mask,\n"
        "instance_mask, intrinsics, extrinsics(camera-to-world OpenCV), state and action.\n",
        encoding="utf-8",
    )
    commit_directory(temporary_root, output_root)
    log(f"[done] {suite_name} -> {output_root}")


def build_lerobot_suites(args: argparse.Namespace) -> None:
    for suite_name in args.suites:
        build_lerobot_suite(args, suite_name)


def validate_outputs(args: argparse.Namespace) -> None:
    failures: list[str] = []
    for suite_name in args.suites:
        suite_failure_count = len(failures)
        root = args.output_dir / output_name(suite_name, args.keep_noops)
        info_path = root / "meta" / "info.json"
        links_path = root / "meta" / "annotations_3d.jsonl"
        if not info_path.exists() or not links_path.exists():
            failures.append(f"{suite_name}: missing LeRobot metadata")
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        links = [json.loads(line) for line in links_path.read_text(encoding="utf-8").splitlines() if line]
        parquet_count = len(list((root / "data").rglob("*.parquet")))
        expected_videos = info["total_episodes"] * 2
        video_count = len(list((root / "videos").rglob("*.mp4")))
        if parquet_count != info["total_episodes"]:
            failures.append(f"{suite_name}: parquet={parquet_count}, episodes={info['total_episodes']}")
        if video_count != expected_videos:
            failures.append(f"{suite_name}: videos={video_count}, expected={expected_videos}")
        if len(links) != info["total_episodes"]:
            failures.append(f"{suite_name}: annotation links={len(links)}, episodes={info['total_episodes']}")
        for row in links:
            annotation_file = root / row["file"]
            if not annotation_file.exists():
                failures.append(f"{suite_name}: missing {annotation_file}")
                break
            with h5py.File(annotation_file, "r") as annotation:
                group = annotation[row["group"]]
                required = {
                    "depth", "robot_mask", "object_mask", "object_of_interest_mask",
                    "instance_mask", "extrinsics", "state", "action",
                }
                missing = required.difference(group.keys())
                if missing or len(group["depth"]) != row["length"]:
                    failures.append(f"{suite_name}: invalid annotation {row}: missing={sorted(missing)}")
                    break
                sample_indices = sorted({0, row["length"] - 1})
                for sample_index in sample_indices:
                    depth = np.asarray(group["depth"][sample_index])
                    robot_mask = np.asarray(group["robot_mask"][sample_index])
                    object_mask = np.asarray(group["object_mask"][sample_index])
                    if not np.all(np.isfinite(depth)) or float(depth.min()) < 0:
                        failures.append(f"{suite_name}: invalid depth in {row}")
                        break
                    if not set(np.unique(robot_mask)).issubset({0, 1}):
                        failures.append(f"{suite_name}: non-binary robot mask in {row}")
                        break
                    if not set(np.unique(object_mask)).issubset({0, 1}):
                        failures.append(f"{suite_name}: non-binary object mask in {row}")
                        break
                    if np.any(robot_mask & object_mask):
                        failures.append(f"{suite_name}: overlapping robot/object masks in {row}")
                        break
                intrinsics = np.asarray(annotation["intrinsics"])
                if intrinsics.shape != (2, 3, 3) or not np.all(np.isfinite(intrinsics)):
                    failures.append(f"{suite_name}: invalid intrinsics in {annotation_file}")
                    break
                camera_index = 1
                depth_values = np.asarray(group["depth"][sample_indices[-1]])
                # robot_mask is stored as (frames, cameras, H, W); select the
                # camera as well so np.nonzero() returns exactly 2 index arrays.
                mask = np.asarray(group["robot_mask"][sample_indices[-1], camera_index]) > 0
                if mask.any():
                    pixel_rows, pixel_cols = np.nonzero(mask)
                    z = depth_values[camera_index, pixel_rows, pixel_cols].astype(np.float64)
                    k_matrix = intrinsics[camera_index].astype(np.float64)
                    extrinsic = np.asarray(group["extrinsics"][sample_indices[-1], camera_index], dtype=np.float64)
                    u = pixel_cols.astype(np.float64)
                    v = pixel_rows.astype(np.float64)
                    z_flattened = z.flatten()
                    ones = np.ones_like(z_flattened)
                    u_flat = u.flatten()
                    v_flat = v.flatten()
                    pixels = np.stack([u_flat, v_flat, ones], axis=0)
                    camera_points = z_flattened[None, :] * (np.linalg.inv(k_matrix) @ pixels)
                    homogeneous = np.vstack([camera_points, ones])
                    world_points = (extrinsic @ homogeneous)[:3]
                    valid = np.isfinite(z_flattened) & (z_flattened > 0)
                    if valid.any():
                        # world -> camera: the extrinsic is a camera-to-world
                        # transform, so its inverse must be applied before the
                        # points can be projected with the intrinsic matrix.
                        world_valid = world_points[:, valid]
                        world_homogeneous = np.vstack(
                            [world_valid, np.ones((1, world_valid.shape[1]))]
                        )
                        camera_back = np.linalg.inv(extrinsic) @ world_homogeneous
                        projected = k_matrix @ (camera_back[:3] / camera_back[2:3])
                        reproject_error = float(
                            np.max(
                                np.linalg.norm(
                                    projected[:2] - np.stack([u_flat[valid], v_flat[valid]]), axis=0
                                )
                            )
                        )
                        if reproject_error > 2.0:
                            failures.append(
                                f"{suite_name}: reprojection error {reproject_error:.2f}px in {row}"
                            )
        if len(failures) == suite_failure_count:
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset

                dataset = LeRobotDataset(f"local/{root.name}", root=root)
                if len(dataset) != info["total_frames"]:
                    failures.append(f"{suite_name}: loader length={len(dataset)}, frames={info['total_frames']}")
                elif len(dataset):
                    for sample_index in sorted({0, len(dataset) - 1}):
                        sample = dataset[sample_index]
                        for key in ("observation.images.image", "observation.images.wrist_image"):
                            if key not in sample or tuple(sample[key].shape[-2:]) != (
                                info["features"][key]["shape"][1],
                                info["features"][key]["shape"][2],
                            ):
                                failures.append(f"{suite_name}: LeRobot decode failed for {key}")
            except Exception as error:
                failures.append(f"{suite_name}: LeRobot load/decode failed: {error}")
        log(
            f"[validate] {suite_name}: episodes={info['total_episodes']}, "
            f"frames={info['total_frames']}, parquet={parquet_count}, videos={video_count}"
        )
    if failures:
        raise RuntimeError("Validation failed:\n- " + "\n- ".join(failures))
    log("[validate] all requested suites passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=Path("/root/dataset/libero_raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/dataset/libero_converted"))
    parser.add_argument("--stage", choices=("render", "lerobot", "all", "validate"), default="all")
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--noop-threshold", type=float, default=1e-4)
    parser.add_argument("--keep-noops", action="store_true")
    parser.add_argument(
        "--replay-mode", choices=("action", "state"), default="state",
        help=(
            "How to drive the renderer. 'state' (default) restores every recorded "
            "MuJoCo state verbatim, so rendered frames, depth and masks match the "
            "source demo exactly regardless of the MuJoCo version. 'action' replays "
            "the recorded actions through the physics engine (this is how "
            "LIBERO-fastwam was regenerated, but it diverges when the local MuJoCo "
            "version differs from the one used to record the demos)."
        ),
    )
    parser.add_argument(
        "--settle-steps", type=int, default=0,
        help="Optional no-op prefix after restoring the initial state; nonzero values change source-state alignment",
    )
    parser.add_argument(
        "--max-state-error", type=float, default=0.05,
        help="Abort action replay when the maximum flattened-state L2 error exceeds this value",
    )
    parser.add_argument(
        "--no-rotate", action="store_true",
        help="Save OpenCV-oriented images without the extra 180-degree rotation used by the FastWAM reference data",
    )
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--keep-failed-replays", action="store_true")
    parser.add_argument("--overwrite-render", action="store_true")
    parser.add_argument("--overwrite-lerobot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.raw_dir = args.raw_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage in ("render", "all"):
            replay_suites(args)
        if args.stage in ("lerobot", "all"):
            build_lerobot_suites(args)
        if args.stage in ("validate", "all"):
            validate_outputs(args)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

PYEOF

cat > "$WORKDIR/requirements-convert.txt" <<'REQEOF'
numpy==1.26.4
h5py>=3.10,<4
mujoco==3.3.2
robosuite==1.4.0
bddl==1.0.1
hydra-core==1.3.2
easydict==1.13
cloudpickle>=2.2,<4
gym==0.25.2
future>=0.18.3
matplotlib>=3.7,<4
PyYAML>=6,<7
huggingface_hub[cli]>=0.34,<1
lerobot==0.3.3
REQEOF

log "Wrote convert_libero_fastwam.py and requirements-convert.txt to $WORKDIR"

# ---- Python environment ----
# NOTE: conda create is intentionally avoided on this host: conda-forge is not
# reachable through the outbound proxy. We build a plain venv from an existing
# CPython interpreter instead and install everything with pip (pypi.org works).
cd "$WORKDIR"
PYBIN=""
for cand in /opt/conda/bin/python3 /usr/bin/python3 "$(command -v python3 || true)"; do
  if [ -n "$cand" ] && [ -x "$cand" ]; then
    if "$cand" -c 'import venv, ensurepip' >/dev/null 2>&1; then PYBIN="$cand"; break; fi
  fi
done
if [ -z "$PYBIN" ]; then
  fail_and_cleanup "no CPython with working venv/ensurepip found"
fi
log "Using base interpreter for venv: $PYBIN ($("$PYBIN" --version 2>&1))"
# --system-site-packages reuses the host's already-installed torch/CUDA stack
# so pip does not have to download a second multi-GB copy of torch.
"$PYBIN" -m venv --system-site-packages "$WORKDIR/venv" >>"$LOG_FILE" 2>&1 || fail_and_cleanup "venv creation failed"
# shellcheck disable=SC1091
source "$WORKDIR/venv/bin/activate"
hash -r
python -V >>"$LOG_FILE" 2>&1
log "venv ready at $WORKDIR/venv"

# pip: use the Aliyun mirror WITHOUT the proxy (measured ~8MB/s vs ~40KB/s
# through the outbound proxy on this host).
PIP_INDEX_ARGS="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
log "pip will use Aliyun mirror (proxy disabled for pip)"

pip install --upgrade pip $PIP_INDEX_ARGS >>"$LOG_FILE" 2>&1

# Plain pip only reaches ~100KB/s from the mirror on this host, while curl gets
# ~8MB/s. So fetch uv with curl and let uv do the heavy installation.
UV_OK=0
UV_WHL=$(python3 - <<'UVEOF' 2>>"$LOG_FILE"
import re, urllib.request
idx = urllib.request.urlopen("https://mirrors.aliyun.com/pypi/simple/uv/", timeout=60).read().decode()
urls = re.findall(r'href="([^"]+)"', idx)
cands = [u for u in urls if "py3-none-manylinux_2_17_x86_64" in u or "py3-none-manylinux2014_x86_64" in u]
print("https://mirrors.aliyun.com/pypi/" + cands[-1].split("#")[0].replace("../../", ""))
UVEOF
)
log "uv wheel url: ${UV_WHL:-<none>}"
UV_WHL_FILE="/tmp/$(basename "${UV_WHL:-uv.whl}")"
if [ -n "$UV_WHL" ]; then
  curl -sS -m 300 -o "$UV_WHL_FILE" "$UV_WHL" >>"$LOG_FILE" 2>&1 || log "WARN: uv wheel download failed"
fi
if [ -s "$UV_WHL_FILE" ]; then
  pip install --no-index "$UV_WHL_FILE" >>"$LOG_FILE" 2>&1 && UV_OK=1
fi
log "uv available: $UV_OK"

if [ "$UV_OK" = "1" ]; then
  uv pip install -r requirements-convert.txt hf_transfer $PIP_INDEX_ARGS >>"$LOG_FILE" 2>&1 \
    || fail_and_cleanup "uv pip install requirements failed (see $LOG_FILE)"
else
  pip install -r requirements-convert.txt hf_transfer $PIP_INDEX_ARGS >>"$LOG_FILE" 2>&1 \
    || fail_and_cleanup "pip install requirements failed (see $LOG_FILE)"
fi

# Fetch the LIBERO source as a zip from a mirror. Direct GitHub access through
# the outbound proxy is unusably slow here (~60KB/s, TLS handshake failures),
# while the mirror serves the 243MB archive at ~9MB/s without any proxy.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

if [ ! -d "$LIBERO_SRC_DIR" ]; then
  log "Downloading LIBERO source zip (pinned commit) via mirror"
  ZIP_URL="https://gh-proxy.com/https://github.com/Lifelong-Robot-Learning/LIBERO/archive/8f1084e3132a39270c3a13ebe37270a43ece2a01.zip"
  curl -sSL -m 900 -o /tmp/libero_src_smoke.zip "$ZIP_URL" >>"$LOG_FILE" 2>&1 \
    || fail_and_cleanup "LIBERO source zip download failed"
  python3 - "$LIBERO_SRC_DIR" >>"$LOG_FILE" 2>&1 <<'PYEOF' || fail_and_cleanup "LIBERO source extraction failed"
import os, shutil, sys, tempfile, zipfile

dest = sys.argv[1]
tmp = tempfile.mkdtemp(prefix="libero_src_")
with zipfile.ZipFile("/tmp/libero_src_smoke.zip") as zf:
    zf.extractall(tmp)
inner = [os.path.join(tmp, d) for d in os.listdir(tmp) if os.path.isdir(os.path.join(tmp, d))]
assert len(inner) == 1, inner
os.makedirs(dest, exist_ok=True)
for item in os.listdir(inner[0]):
    shutil.move(os.path.join(inner[0], item), os.path.join(dest, item))
shutil.rmtree(tmp, ignore_errors=True)
print("extracted LIBERO to", dest)
PYEOF
  rm -f /tmp/libero_src_smoke.zip
  touch "$LIBERO_SRC_DIR/$MARKER"
  CREATED_DIRS+=("$LIBERO_SRC_DIR")
  log "LIBERO source ready at $LIBERO_SRC_DIR"
fi
# The zip is already pinned to commit 8f1084e, so no checkout is needed.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
if [ "$UV_OK" = "1" ]; then
  uv pip install -e "$LIBERO_SRC_DIR" $PIP_INDEX_ARGS >>"$LOG_FILE" 2>&1 || fail_and_cleanup "uv pip install -e LIBERO failed"
else
  pip install -e "$LIBERO_SRC_DIR" $PIP_INDEX_ARGS >>"$LOG_FILE" 2>&1 || fail_and_cleanup "pip install -e LIBERO failed"
fi

LIBERO_CONFIG_CREATED=0
mkdir -p /root/.libero
if [ ! -f /root/.libero/config.yaml ]; then
  cat > /root/.libero/config.yaml <<YAMLEOF
benchmark_root: $LIBERO_SRC_DIR/libero/libero
bddl_files: $LIBERO_SRC_DIR/libero/libero/bddl_files
init_states: $LIBERO_SRC_DIR/libero/libero/init_files
datasets: $DATA_RAW_DIR
assets: $LIBERO_SRC_DIR/libero/libero/assets
YAMLEOF
  LIBERO_CONFIG_CREATED=1
fi

# ---- GPU selection: pick one idle GPU automatically ----
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ID=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | sort -t',' -k2 -n | head -n1 | cut -d',' -f1 | tr -d ' ')
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  export MUJOCO_EGL_DEVICE_ID="$GPU_ID"
  export MUJOCO_GL=egl
  log "Selected idle GPU index: $GPU_ID (CUDA_VISIBLE_DEVICES=$GPU_ID)"
else
  log "No nvidia-smi found, falling back to MUJOCO_GL=osmesa (CPU rendering)"
  export MUJOCO_GL=osmesa
fi

# ---- Download exactly ONE raw task HDF5 file ----
# HuggingFace needs the outbound proxy again
if [ -f /root/clashctl/scripts/cmd/clashctl.sh ]; then
  export http_proxy="http://127.0.0.1:7890"
  export https_proxy="http://127.0.0.1:7890"
  export HTTP_PROXY="$http_proxy"
  export HTTPS_PROXY="$https_proxy"
  export no_proxy="localhost,127.0.0.1,::1"
  export NO_PROXY="$no_proxy"
  log "Proxy re-enabled for HuggingFace download"
fi
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$DATA_RAW_DIR"
TASK_FILE=$(python3 - <<PYEOF2
from huggingface_hub import list_repo_files
files = [f for f in list_repo_files("yifengzhu-hf/LIBERO-datasets", repo_type="dataset")
         if f.startswith("$SUITE/") and f.endswith(".hdf5")]
print(sorted(files)[0])
PYEOF2
) || fail_and_cleanup "listing HF repo files failed (network/auth issue?)"
log "Selected task file for smoke test: $TASK_FILE"

# The proxy randomly stalls mid-transfer, so run the resumable download in a
# bounded retry loop instead of trusting a single long-lived connection.
DOWNLOAD_OK=0
for attempt in 1 2 3 4 5 6 7 8; do
  if [ -f "$DATA_RAW_DIR/$TASK_FILE" ]; then DOWNLOAD_OK=1; break; fi
  log "HF download attempt $attempt (timeout 600s, resumable)"
  timeout 600 huggingface-cli download yifengzhu-hf/LIBERO-datasets \
    --repo-type dataset \
    --include "$TASK_FILE" \
    --local-dir "$DATA_RAW_DIR" >>"$LOG_FILE" 2>&1
  if [ -f "$DATA_RAW_DIR/$TASK_FILE" ]; then DOWNLOAD_OK=1; break; fi
done
if [ "$DOWNLOAD_OK" != "1" ]; then
  fail_and_cleanup "huggingface-cli download failed after retries"
fi

DOWNLOADED_FILES+=("$DATA_RAW_DIR/$TASK_FILE")
log "Downloaded: $DATA_RAW_DIR/$TASK_FILE"

# ---- Run smoke conversion: 1 task, 1 demo, 20 frames ----
set +e
python convert_libero_fastwam.py \
  --raw-dir "$DATA_RAW_DIR" \
  --output-dir "$SMOKE_OUT_DIR" \
  --suites "$SUITE" \
  --max-tasks 1 \
  --max-demos 1 \
  --max-frames 20 \
  --keep-failed-replays \
  --stage all 2>&1 | tee -a "$LOG_FILE"
CONVERT_STATUS=${PIPESTATUS[0]}
set -e

if [ -d "$SMOKE_OUT_DIR" ]; then
  touch "$SMOKE_OUT_DIR/$MARKER"
  CREATED_DIRS+=("$SMOKE_OUT_DIR")
fi
if [ "$CONVERT_STATUS" -ne 0 ]; then
  fail_and_cleanup "conversion script exited with status $CONVERT_STATUS"
fi

# ---- Validate ----
set +e
python convert_libero_fastwam.py \
  --stage validate \
  --output-dir "$SMOKE_OUT_DIR" \
  --suites "$SUITE" 2>&1 | tee -a "$LOG_FILE"
VALIDATE_STATUS=${PIPESTATUS[0]}
set -e
if [ "$VALIDATE_STATUS" -ne 0 ]; then
  fail_and_cleanup "validation failed with status $VALIDATE_STATUS"
fi

log "=========================================="
log "SMOKE TEST PASSED."
log "Output kept at:            $SMOKE_OUT_DIR"
log "Script/venv kept at:       $WORKDIR"
log "LIBERO source kept at:     $LIBERO_SRC_DIR"
log "Downloaded raw file kept:  $DATA_RAW_DIR/$TASK_FILE"
log "=========================================="
log "Nothing was deleted since the test succeeded. Full log: $LOG_FILE"
