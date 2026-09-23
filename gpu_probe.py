"""Minimal EGL probe: exit 0 only if offscreen rendering actually works."""

import mujoco

model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type="box" size="1 1 1"/></worldbody></mujoco>'
)
renderer = mujoco.Renderer(model, 64, 64)
renderer.render()
print("EGL_OK")
