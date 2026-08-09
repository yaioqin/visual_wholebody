"""Headless GPU camera replay for the Aliengo-Z1 low-level policy.

The script keeps Isaac Gym graphics enabled without creating an interactive
viewer, selects the NVIDIA EGL Vulkan ICD, and overlays the VBC debug markers
onto the camera RGB frames before writing an MP4.
"""

from pathlib import Path
import os
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
LOW_LEVEL_DIR = SCRIPT_DIR.parents[1]
REPO_DIR = LOW_LEVEL_DIR.parent
ISAACGYM_DIR = REPO_DIR / "third_party" / "isaacgym" / "python"
RSL_RL_DIR = REPO_DIR / "third_party" / "rsl_rl"
for import_path in (str(ISAACGYM_DIR), str(RSL_RL_DIR), str(LOW_LEVEL_DIR), str(SCRIPT_DIR)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)
os.chdir(SCRIPT_DIR)


EGL_ICD = Path(os.environ.get("VBC_EGL_ICD", "/tmp/nvidia_icd_egl.json"))
if not EGL_ICD.exists():
    EGL_ICD.write_text(
        '{\n'
        '  "file_format_version": "1.0.1",\n'
        '  "ICD": {\n'
        '    "library_path": "libEGL_nvidia.so.0",\n'
        '    "api_version": "1.4.312"\n'
        '  }\n'
        '}\n'
    )
os.environ.setdefault("VK_ICD_FILENAMES", str(EGL_ICD))
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")
Path(os.environ["XDG_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
from PIL import Image, ImageDraw
from isaacgym import gymapi
import play
from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from legged_gym.utils import task_registry


def normalize(vector):
    return vector / max(np.linalg.norm(vector), 1e-8)


def yaw_matrix(quaternion):
    x, y, z, w = quaternion
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def quaternion_matrix(quaternion):
    x, y, z, w = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def projector(camera_position, camera_target, width, height):
    forward = normalize(camera_target - camera_position)
    right = normalize(np.cross(forward, np.array([0.0, 0.0, 1.0])))
    up = normalize(np.cross(right, forward))
    focal = width / (2.0 * np.tan(np.deg2rad(75.0) / 2.0))

    def project(point):
        delta = np.asarray(point) - camera_position
        depth = np.dot(delta, forward)
        if depth <= 0.05:
            return None
        return (
            float(width * 0.5 + focal * np.dot(delta, right) / depth),
            float(height * 0.5 - focal * np.dot(delta, up) / depth),
        )

    return project


def ring(draw, point, radius, color):
    if point is None:
        return
    x, y = point
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
    draw.line((x - radius - 2, y, x + radius + 2, y), fill=color, width=2)
    draw.line((x, y - radius - 2, x, y + radius + 2), fill=color, width=2)


def overlay(env, env_index, rgba, camera_position, camera_target):
    height, width = rgba.shape[:2]
    canvas = Image.fromarray(rgba, mode="RGBA")
    draw = ImageDraw.Draw(canvas)
    project = projector(camera_position, camera_target, width, height)
    root = env.root_states[env_index, :3].detach().cpu().numpy()
    rotation = yaw_matrix(env.base_quat[env_index].detach().cpu().numpy())
    actual = env.ee_pos[env_index].detach().cpu().numpy()
    target = env.curr_ee_goal_cart_world[env_index].detach().cpu().numpy()
    center = env._get_ee_goal_spherical_center()[env_index].detach().cpu().numpy()
    ring(draw, project(actual), 8, (20, 80, 255, 255))
    ring(draw, project(target), 9, (255, 230, 0, 255))
    ring(draw, project(center), 7, (0, 220, 255, 255))

    start = env.ee_start_sphere[env_index].detach().cpu().numpy()
    goal = env.ee_goal_sphere[env_index].detach().cpu().numpy()
    for alpha in np.linspace(0.0, 1.0, 10):
        length, pitch, yaw = start * (1.0 - alpha) + goal * alpha
        local = np.array([
            length * np.cos(pitch) * np.cos(yaw),
            length * np.cos(pitch) * np.sin(yaw),
            length * np.sin(pitch),
        ])
        point = project(center + rotation @ local)
        if point is not None:
            x, y = point
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 35, 25, 255))

    center_local = env.ee_goal_center_offset[env_index].detach().cpu().numpy()
    lower = center_local + env.collision_lower_limits.detach().cpu().numpy()
    upper = center_local + env.collision_upper_limits.detach().cpu().numpy()
    minimum, maximum = np.minimum(lower, upper), np.maximum(lower, upper)
    local_corners = np.array([
        [maximum[0] if bits & 1 else minimum[0],
         maximum[1] if bits & 2 else minimum[1],
         maximum[2] if bits & 4 else minimum[2]]
        for bits in range(8)
    ])
    corners = [project(np.array([root[0], root[1], 0.0]) + rotation @ corner) for corner in local_corners]
    for first in range(8):
        for bit in (1, 2, 4):
            second = first ^ bit
            if first < second and corners[first] is not None and corners[second] is not None:
                draw.line((*corners[first], *corners[second]), fill=(255, 30, 20, 255), width=3)

    target_rotation = quaternion_matrix(env.ee_goal_orn_quat[env_index].detach().cpu().numpy())
    for axis, color in zip(np.eye(3), ((255, 30, 20, 255), (20, 220, 40, 255), (30, 100, 255, 255))):
        start_pixel = project(target)
        end_pixel = project(target + target_rotation @ (axis * 0.18))
        if start_pixel is not None and end_pixel is not None:
            draw.line((*start_pixel, *end_pixel), fill=color, width=3)
    return np.asarray(canvas).copy()


def render_record(env, mode="rgb_array"):
    if env.global_steps % 2 != 0:
        return None
    env.gym.fetch_results(env.sim, True)
    cameras = []
    for index, camera in enumerate(env._rendering_camera_handles):
        root_world = env.root_states[index, :3].detach().cpu().numpy()
        camera_position = root_world + np.array([-0.9, 1.6, 0.65])
        camera_target = root_world + np.array([0.2, 0.0, 0.12])
        cameras.append((camera_position, camera_target))
        env.gym.set_camera_location(
            camera, env.envs[index], gymapi.Vec3(*camera_position), gymapi.Vec3(*camera_target)
        )
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    images = []
    for index, camera in enumerate(env._rendering_camera_handles):
        image = env.gym.get_camera_image(env.sim, env.envs[index], camera, gymapi.IMAGE_COLOR)
        height, packed_width = image.shape
        rgba = image.reshape(height, packed_width // 4, 4)
        images.append(overlay(env, index, rgba, *cameras[index]))
    return images


ManipLoco.render_record = render_record
play.EXPORT_POLICY = False
play.SAVE_ACTOR_HIST_ENCODER = False
args = play.get_args()
# True disables graphics in BaseTask. This sentinel is neither True nor False,
# so Isaac Gym keeps the camera graphics device without creating a viewer.
args.headless = object()
play.play(args)
