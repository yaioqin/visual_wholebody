"""Temporary replay: move the EE position target continuously in world +X.

This is an evaluation-only probe. It does not change the training config or
the checkpoint. The target starts at the reset target and moves along the
initial base-forward direction, while the policy and autonomous base commands
remain unchanged.
"""

from pathlib import Path
import os
import sys
import time
from types import MethodType


SCRIPT_DIR = Path(__file__).resolve().parent
LOW_LEVEL_DIR = SCRIPT_DIR.parents[1]
REPO_DIR = LOW_LEVEL_DIR.parent
ISAACGYM_DIR = REPO_DIR / "third_party" / "isaacgym" / "python"
RSL_RL_DIR = REPO_DIR / "third_party" / "rsl_rl"
for import_path in (str(ISAACGYM_DIR), str(RSL_RL_DIR), str(LOW_LEVEL_DIR), str(SCRIPT_DIR)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)
os.chdir(SCRIPT_DIR)


# Keep graphics available for Isaac Gym camera rendering without Xorg.
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

import imageio
import numpy as np
import play
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse
from PIL import Image, ImageDraw
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from legged_gym.utils import task_registry
from legged_gym.utils.math import cart2sphere


BASE_SPEED_MPS = 0.80
TARGET_SPEED_MPS = 0.80
INITIAL_TARGET_LEAD_M = 1.00
PLAY_SECONDS = 12.0
VIDEO_TAG = "forward-far-fast"


def normalize(vector):
    return vector / max(np.linalg.norm(vector), 1e-8)


def make_projector(camera_position, camera_target, width, height):
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


def draw_marker(draw, point, radius, color, label):
    if point is None:
        return
    x, y = point
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
    draw.line((x - radius - 2, y, x + radius + 2, y), fill=color, width=2)
    draw.line((x, y - radius - 2, x, y + radius + 2), fill=color, width=2)
    draw.text((x + radius + 4, y - 10), label, fill=color)


def overlay_target_markers(env, env_index, rgba, camera_position, camera_target):
    height, width = rgba.shape[:2]
    canvas = Image.fromarray(rgba, mode="RGBA")
    draw = ImageDraw.Draw(canvas)
    project = make_projector(camera_position, camera_target, width, height)
    current = env.ee_pos[env_index].detach().cpu().numpy()
    target = env.curr_ee_goal_cart_world[env_index].detach().cpu().numpy()
    draw_marker(draw, project(current), 8, (20, 80, 255, 255), "current")
    draw_marker(draw, project(target), 9, (255, 220, 0, 255), "target")
    return np.asarray(canvas).copy()


def install_forward_target(env):
    """Replace only target updates for this replay instance."""
    original_update = env._update_curr_ee_goal
    original_update()

    initial_forward = quat_apply(
        env.base_yaw_quat.detach(),
        torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1),
    )
    target_anchor = env.curr_ee_goal_cart_world.detach().clone()
    target_anchor += initial_forward * INITIAL_TARGET_LEAD_M
    state = {"steps": 0}

    def force_forward_command(self, env_ids):
        self.commands[env_ids, 0] = BASE_SPEED_MPS
        self.commands[env_ids, 1] = 0.0
        self.commands[env_ids, 2] = 0.0

    def update_forward(self):
        distance = TARGET_SPEED_MPS * state["steps"] * self.dt
        self.curr_ee_goal_cart_world[:] = target_anchor + initial_forward * distance

        # Keep local/spherical target buffers coherent for observations and
        # debug rendering while the world target moves independently of base.
        center = self._get_ee_goal_spherical_center()
        local_target = quat_rotate_inverse(
            self.base_yaw_quat,
            self.curr_ee_goal_cart_world - center,
        )
        self.curr_ee_goal_cart[:] = local_target
        self.curr_ee_goal_sphere[:] = cart2sphere(local_target)
        self.goal_timer += 1
        state["steps"] += 1

    env._resample_commands = MethodType(force_forward_command, env)
    env._resample_commands(torch.arange(env.num_envs, device=env.device))
    env._update_curr_ee_goal = MethodType(update_forward, env)
    env._update_curr_ee_goal()
    env.compute_observations()
    return target_anchor, initial_forward


def render_follow_camera(self, mode="rgb_array"):
    """Render a camera in world coordinates while following the robot root."""
    if self.global_steps % 2 != 0:
        return None
    self.gym.fetch_results(self.sim, True)
    camera_data = []
    for index, camera in enumerate(self._rendering_camera_handles):
        root = self.root_states[index, :3].detach().cpu().numpy()
        target = self.curr_ee_goal_cart_world[index].detach().cpu().numpy()
        target_delta = target - root
        planar_distance = max(np.linalg.norm(target_delta[:2]), 1e-3)
        forward = target_delta.copy()
        forward[2] = 0.0
        forward = normalize(forward)
        side = np.array([-forward[1], forward[0], 0.0])
        midpoint = 0.5 * (root + target)
        camera_position = (
            midpoint
            - forward * (1.0 + 0.7 * planar_distance)
            + side * (0.8 + 0.5 * planar_distance)
            + np.array([0.0, 0.0, 0.8 + 0.2 * planar_distance])
        )
        camera_target = midpoint + np.array([0.0, 0.0, 0.15])
        self.gym.set_camera_location(
            camera,
            self.envs[index],
            gymapi.Vec3(*camera_position),
            gymapi.Vec3(*camera_target),
        )
        camera_data.append((camera, camera_position, camera_target))

    self.gym.step_graphics(self.sim)
    self.gym.render_all_camera_sensors(self.sim)
    images = []
    for index, (camera, _, _) in enumerate(camera_data):
        image = self.gym.get_camera_image(
            self.sim,
            self.envs[index],
            camera,
            gymapi.IMAGE_COLOR,
        )
        height, packed_width = image.shape
        frame = image.reshape(height, packed_width // 4, 4)
        _, camera_position, camera_target = camera_data[index]
        images.append(overlay_target_markers(
            self,
            index,
            frame,
            camera_position,
            camera_target,
        ))
    return images


def play_forward(args):
    log_root = Path(LEGGED_GYM_ROOT_DIR) / "logs" / args.proj_name / args.exptid
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 6
    env_cfg.terrain.num_cols = 3
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.randomize_base_com = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    runner, train_cfg, checkpoint, _ = task_registry.make_alg_runner(
        log_root=str(log_root),
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        return_log_dir=True,
    )
    policy = runner.get_inference_policy(device=env.device, stochastic=args.stochastic)

    env.reset()
    target_anchor, initial_forward = install_forward_target(env)
    env.render_record = MethodType(render_follow_camera, env)
    obs = env.get_observations()

    env.enable_viewer_sync = False
    video_dir = Path(LEGGED_GYM_ROOT_DIR) / "logs" / "videos" / args.exptid
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"{args.exptid}-{VIDEO_TAG}-0-{checkpoint}.mp4"
    writer = imageio.get_writer(str(video_path), fps=25)

    steps = int(PLAY_SECONDS / env.dt)
    max_leg_action = 0.0
    try:
        for step in range(steps):
            start_time = time.time()
            if args.use_jit:
                policy_obs = torch.cat(
                    (obs[:, :env.cfg.env.num_proprio],
                     obs[:, env.cfg.env.num_proprio + env.cfg.env.num_priv:]),
                    dim=1,
                )
                actions = policy(policy_obs)
            else:
                actions = policy(obs.detach(), hist_encoding=True)

            max_leg_action = max(max_leg_action, actions[0, :12].abs().max().item())
            obs, _, _, _, _, _ = env.step(actions.detach())
            image = env.render_record(mode="rgb_array")
            if image is not None:
                writer.append_data(image[0])

            if step % 50 == 0:
                target = env.curr_ee_goal_cart_world[0].detach().cpu().numpy()
                root = env.root_states[0, :3].detach().cpu().numpy()
                print(
                    "step", step,
                    "target", np.round(target, 3),
                    "root", np.round(root, 3),
                    "cmd", np.round(env.commands[0, :3].detach().cpu().numpy(), 3),
                    "lin", np.round(env.base_lin_vel[0, :3].detach().cpu().numpy(), 3),
                    "act_leg_abs", round(actions[0, :12].abs().mean().item(), 4),
                    "tau_leg_abs", round(env.torques[0, :12].abs().mean().item(), 4),
                )

            time.sleep(max(0.02 - (time.time() - start_time), 0.0))
    finally:
        writer.close()

    print("Forward-target replay:", video_path)
    print("base_speed_mps:", BASE_SPEED_MPS)
    print("target_speed_mps:", TARGET_SPEED_MPS)
    print("initial_target_lead_m:", INITIAL_TARGET_LEAD_M)
    print("max_leg_action_abs:", max_leg_action)


if __name__ == "__main__":
    play.EXPORT_POLICY = False
    play.SAVE_ACTOR_HIST_ENCODER = False
    args = play.get_args()
    args.headless = object()
    args.record_video = True
    play_forward(args)
