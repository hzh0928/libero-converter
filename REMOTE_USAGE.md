# LIBERO 原始数据回放与 FastWAM 格式转换

## 输出内容

脚本生成四个与 `yuanty/LIBERO-fastwam` 同名、兼容 LeRobot v2.1 的目录：

```text
libero_converted/
├── libero_spatial_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
└── libero_10_no_noops_lerobot/
```

每个目录中的标准字段与参考数据一致：

- `observation.images.image`：`agentview` RGB 视频；
- `observation.images.wrist_image`：腕部 RGB 视频；
- `observation.state`：6 维末端状态 + 2 维夹爪状态；
- `observation.states.ee_state`、`joint_state`、`gripper_state`；
- `action`：7 维，夹爪被转换为 FastWAM 的 `1=open, 0=close`；
- 标准 `timestamp/frame_index/episode_index/index/task_index`。

参考数据本身没有深度、mask 和相机参数。为保证深度不被视频有损压缩，脚本在每个 LeRobot suite 内增加 HDF5 sidecar：

```text
<dataset>/
├── data/
├── meta/
│   └── annotations_3d.jsonl
├── videos/
└── annotations_3d/
    └── tasks/*.hdf5
```

sidecar 字段：

- `rgb`：两相机 RGB；
- `depth`：米制 z-depth，`float16`；
- `robot_mask`：机器人二值 mask；
- `object_mask`：所有非机器人实例的并集；
- `object_of_interest_mask`：BDDL 任务关注物体的并集；
- `instance_mask`：原始 instance ID，映射在 HDF5 根属性 `instance_id_to_name`；
- `intrinsics`：每相机 `3x3` 内参；
- `extrinsics`：逐帧 `4x4` camera-to-world，OpenCV 相机坐标系；
- `source_state`、`replayed_sim_state`、机器人状态、原始与转换后 action。

`meta/annotations_3d.jsonl` 将 LeRobot `episode_index` 映射到 HDF5 文件及 group。

## 1. 本地上传脚本到远程电脑

在本机终端执行，将占位符替换为远程用户名和地址：

```bash
ssh USER@REMOTE_HOST 'mkdir -p /root/libero_converter'
scp convert_libero_fastwam.py requirements-convert.txt REMOTE_USAGE.md \
  USER@REMOTE_HOST:/root/libero_converter/
```

如果远程 SSH 不是 22 端口，对 `ssh` 使用 `-p PORT`，对 `scp` 使用 `-P PORT`。

## 2. 在远程电脑安装环境

建议使用 Linux、NVIDIA GPU、Conda Python 3.10。不要在已有 FastWAM 训练环境中直接覆盖依赖。

```bash
conda create -n libero-convert python=3.10 -y
conda activate libero-convert

sudo apt-get update
sudo apt-get install -y git ffmpeg libgl1 libglfw3 libosmesa6-dev

cd /root/libero_converter
pip install -r requirements-convert.txt

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /root/LIBERO
git -C /root/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
pip install -e /root/LIBERO

# 避免 LIBERO 首次 import 时弹出交互式路径询问
mkdir -p /root/.libero
cat >/root/.libero/config.yaml <<'YAML'
benchmark_root: /root/LIBERO/libero/libero
bddl_files: /root/LIBERO/libero/libero/bddl_files
init_states: /root/LIBERO/libero/libero/init_files
datasets: /root/dataset/libero_raw
assets: /root/LIBERO/libero/libero/assets
YAML
```

最后确认版本：

```bash
python - <<'PY'
import mujoco, robosuite
import libero
print('mujoco:', mujoco.__version__)
print('robosuite:', robosuite.__version__)
print('libero:', libero.__file__)
PY
```

参考数据声明使用 MuJoCo `3.3.2`；若远程现有 LIBERO 环境与该版本不兼容，请优先使用你实际生成原数据时的 LIBERO/robosuite 版本，但不要改变数据对应的 MuJoCo 版本。

## 3. 下载原始数据

你已有的 `/root/dataset/libero` 是参考格式数据，请保留。把原始 HDF5 下载到另一个目录：

```bash
mkdir -p /root/dataset/libero_raw
huggingface-cli download yifengzhu-hf/LIBERO-datasets \
  --repo-type dataset \
  --include 'libero_spatial/**' \
  --include 'libero_object/**' \
  --include 'libero_goal/**' \
  --include 'libero_10/**' \
  --local-dir /root/dataset/libero_raw
```

若服务器不能访问 `huggingface.co`：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

然后重新执行下载命令。检查目录：

```bash
find /root/dataset/libero_raw -name '*_demo.hdf5' | wc -l
```

四个 suite 应为 40 个任务 HDF5。`libero_90` 不属于 FastWAM 参考数据的这四个目录，默认不下载也不转换。

## 4. 先做小规模测试

`--max-frames` 会提前截断轨迹，因此测试时加 `--keep-failed-replays`：

```bash
cd /root/libero_converter
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python convert_libero_fastwam.py \
  --raw-dir /root/dataset/libero_raw \
  --output-dir /root/dataset/libero_smoke \
  --suites libero_spatial \
  --max-tasks 1 \
  --max-demos 1 \
  --max-frames 20 \
  --keep-failed-replays
```

测试成功后验证：

```bash
python convert_libero_fastwam.py \
  --stage validate \
  --output-dir /root/dataset/libero_smoke \
  --suites libero_spatial
```

如果 EGL 初始化失败，可先检查：

```bash
nvidia-smi
MUJOCO_GL=osmesa python -c 'import mujoco; print(mujoco.__version__)'
```

没有 NVIDIA/EGL 时可把运行命令中的 `MUJOCO_GL=egl` 改为 `MUJOCO_GL=osmesa`，但 CPU 渲染会明显更慢。

## 5. 转换完整数据

建议用 `tmux` 或 `nohup`。默认行为是：512×512、20 FPS、两相机、按全部原始动作顺序回放、只在写出时去 no-op、过滤完整但失败的回放。图像方向已通过 `robosuite.macros.IMAGE_CONVENTION = "opencv"` 统一为 OpenCV 方向，默认再做 180° 旋转以匹配 FastWAM 参考数据的视觉方向；该旋转同步应用于深度和 mask，且内参、外参做了一致的数学变换，因此 depth + 内外参反投影点云是自洽的（针对 LIBERO issue #101 的修复）。默认不插入额外稳定步，从而保持原始 state/observation/action 对齐。字段和目录与 FastWAM 参考格式兼容。

```bash
cd /root/libero_converter
mkdir -p /root/dataset/libero_converted
nohup env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
  python convert_libero_fastwam.py \
    --raw-dir /root/dataset/libero_raw \
    --output-dir /root/dataset/libero_converted \
    --stage all \
  > /root/dataset/libero_converted/convert.log 2>&1 &

echo $!
tail -f /root/dataset/libero_converted/convert.log
```

转换分两阶段：

1. `render`：LIBERO 回放并写入可断点续跑的任务级 HDF5；只有输入和参数指纹一致时才跳过。
2. `lerobot`：将成功轨迹写成 LeRobot v2.1 的 Parquet + MP4，并用硬链接（失败时复制）加入 3D sidecar；staging 会保留，方便安全重建。

可分别运行：

```bash
MUJOCO_GL=egl python convert_libero_fastwam.py \
  --stage render \
  --raw-dir /root/dataset/libero_raw \
  --output-dir /root/dataset/libero_converted

python convert_libero_fastwam.py \
  --stage lerobot \
  --output-dir /root/dataset/libero_converted
```

最后完整验证：

```bash
python convert_libero_fastwam.py \
  --stage validate \
  --output-dir /root/dataset/libero_converted
```

## 6. 常用选项

```text
--suites libero_spatial libero_object   只处理指定 suite
--resolution 224                        降低分辨率和存储占用
--replay-mode action                    按全部动作回放，仅过滤写出帧；默认
--replay-mode state                     逐帧恢复记录的 MuJoCo state，避免动作累积误差
--settle-steps N                        恢复初态后额外执行 N 个 no-op；默认 0
--max-state-error 0.05                  动作回放状态 L2 误差超过阈值即停止
--keep-noops                            不删除 no-op
--keep-failed-replays                   将回放失败轨迹也写入 LeRobot
--no-rotate                             不执行参考流程中的 180° 图像旋转
--overwrite-render                      重做已完成的任务级 HDF5
--overwrite-lerobot                     重建已存在的 LeRobot suite
```

512×512 双相机的深度和分割 sidecar 可能占用数百 GB。先运行小测试并用 `du -sh /root/dataset/libero_smoke` 估算总空间；如果不要求与参考数据的原始分辨率一致，建议使用 `--resolution 224` 或 `256`。

## 7. 读取某个 episode 的深度和标定

```python
import json
from pathlib import Path
import h5py

root = Path('/root/dataset/libero_converted/libero_spatial_no_noops_lerobot')
row = json.loads((root / 'meta/annotations_3d.jsonl').read_text().splitlines()[0])
with h5py.File(root / row['file'], 'r') as f:
    g = f[row['group']]
    depth = g['depth'][0]              # (2, H, W), meter
    robot_mask = g['robot_mask'][0]    # (2, H, W), uint8
    object_mask = g['object_mask'][0]  # (2, H, W), uint8
    K = f['intrinsics'][()]             # (2, 3, 3)
    camera_to_world = g['extrinsics'][0]  # (2, 4, 4)
    camera_names = json.loads(f.attrs['camera_names'])
```

点反投影约定：对像素 `(u, v)` 和深度 `z`，先计算 `p_cam = z * inv(K) @ [u, v, 1]`，再计算 `p_world = camera_to_world @ [p_cam, 1]`。

该约定与保存的图像方向、内参、外参是自洽的：`validate` 阶段会对 sidecar 抽样做“深度反投影到世界坐标，再用同一相机投影回像素”的往返检查，往返误差大于 2 像素会直接判定验证失败。若仍怀疑方向问题，可先用 `--stage render --max-frames 20` 生成少量数据并单独跑 `--stage validate` 复核。
