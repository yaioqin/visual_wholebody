"""Exercise production camera methods with a fake Gym, without CUDA/Isaac Gym."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from legged_gym.video import overlay_goal, project_world_points, recording_schedule


ROOT = Path(__file__).resolve().parents[1]


def load_method(path, class_name, method_name, namespace):
    # Avoid importing envs/__init__.py, which loads Isaac Gym native bindings.
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method_name]


class RootStates:
    def __init__(self, positions):
        self.positions = np.array(positions, dtype=float)

    def __getitem__(self, key):
        return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: self.positions[key]))


class VideoRecordingTest(unittest.TestCase):
    def test_headless_recording_keeps_graphics_device(self):
        for headless, record, expected in [(True, True, 2), (True, False, -1), (False, False, 2)]:
            with self.subTest(headless=headless, record=record):
                gym = Mock()
                init = load_method("envs/base/base_task.py", "BaseTask", "__init__", dict(
                    gymapi=SimpleNamespace(acquire_gym=lambda: gym, CameraProperties=Mock()),
                    gymutil=SimpleNamespace(parse_device_str=lambda _: ('cuda', 2)),
                    torch=Mock(),
                ))
                cfg = SimpleNamespace(env=SimpleNamespace(
                    record_video=record, num_envs=1, num_observations=1,
                    num_privileged_obs=None, num_actions=1,
                ))
                env = SimpleNamespace(create_sim=Mock(), sim=object(), subscribe_viewer_keyboard_events=Mock())
                env.create_sim.side_effect = lambda: self.assertEqual(env.graphics_device_id, expected)
                init(env, cfg, SimpleNamespace(use_gpu_pipeline=True), None, 'cuda:2', headless)
                self.assertEqual(env.graphics_device_id, expected)
                self.assertEqual(gym.create_viewer.call_count, int(not headless))

    def test_first_frame_and_following_frames_target_current_robot(self):
        for device in ['cuda:0', 'cpu']:
            with self.subTest(device=device):
                gym = Mock()
                gym.get_env_origin.side_effect = lambda env: SimpleNamespace(x=10 * env, y=20 * env, z=env)
                rgba = np.arange(32, dtype=np.uint8).reshape(2, 16)
                gym.get_camera_image.return_value = rgba
                render = load_method("envs/manip_loco/manip_loco.py", "ManipLoco", "render_record", dict(
                    np=np, gymapi=SimpleNamespace(Vec3=lambda *xyz: xyz, IMAGE_COLOR=0),
                    overlay_goal=Mock(side_effect=lambda rgb, *args: rgb),
                ))
                trajectory = Mock()
                trajectory.detach.return_value.cpu.return_value.numpy.return_value = np.zeros((2, 64, 3))
                goals = Mock()
                goals.detach.return_value.cpu.return_value.numpy.return_value = np.zeros((2, 3))
                env = SimpleNamespace(
                    gym=gym, sim=object(), device=device, global_steps=2, num_envs=2,
                    envs=[0, 1], _rendering_camera_handles=[100, 101],
                    _get_ee_goal_trajectory=Mock(return_value=trajectory), curr_ee_goal_cart_world=goals,
                )
                # Include a large reset jump to catch rendering with the previous camera pose.
                for positions in [[[25, 8, 0.6], [30, 40, 1.6]], [[80, -50, 0.6], [11, 22, 1.6]]]:
                    env.root_states = RootStates(positions)
                    gym.reset_mock()
                    images = render(env)
                    names = [call[0] for call in gym.mock_calls]
                    self.assertEqual(names.count('fetch_results'), int(device != 'cpu'))
                    if device != 'cpu':
                        self.assertLess(names.index('fetch_results'), names.index('step_graphics'))
                    self.assertLess(max(i for i, n in enumerate(names) if n == 'set_camera_location'),
                                    names.index('render_all_camera_sensors'))
                    self.assertLess(names.index('render_all_camera_sensors'), names.index('get_camera_image'))
                    for i, call in enumerate(gym.set_camera_location.call_args_list):
                        cam, handle, position, target = call.args
                        expected = np.array(positions[i]) - np.array([10 * i, 20 * i, i])
                        self.assertEqual((cam, handle), (100 + i, i))
                        np.testing.assert_allclose(target, expected)
                        np.testing.assert_allclose(position, expected + [0, 3, 1.5])
                        np.testing.assert_array_equal(images[i], rgba.reshape(2, 4, 4)[:, :, :3])
                        self.assertTrue(images[i].flags.c_contiguous)

                gym.reset_mock()
                env.global_steps = 3
                self.assertIsNone(render(env))
                self.assertEqual(gym.mock_calls, [])

    def test_sixty_second_recording_spans_multiple_episodes(self):
        for dt in [0.02, 0.01, 0.025]:
            fps, steps = recording_schedule(dt)
            for starting_step in [0, 1, 500]:
                frames = sum(step % 2 == 0 for step in range(starting_step + 1, starting_step + steps + 1))
                self.assertAlmostEqual(frames / fps, 60.0)
            self.assertGreater(steps, int(10 / dt))
        self.assertEqual(recording_schedule(0.02, 120), (25, 6000))
        for duration in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                recording_schedule(0.02, duration)

    @staticmethod
    def camera_matrices():
        # Camera at world (10, 20, 5), looking along -Z; 90-degree horizontal FOV.
        view = np.eye(4)
        view[3, :3] = [-10, -20, -5]
        projection = np.zeros((4, 4))
        projection[0, 0], projection[1, 1] = 1, 4 / 3
        projection[2, 2], projection[2, 3], projection[3, 2] = -1, -1, -0.02
        return view, projection

    def test_projection_world_translation_axes_and_behind_camera(self):
        view, projection = self.camera_matrices()
        points = [[10, 20, 3], [11, 20, 3], [10, 20.75, 3],
                  [10, 20, 6], [10, 20, 5], [100, 20, 3], [float('nan'), 20, 3]]
        pixels, valid = project_world_points(points, view, projection, 640, 480)
        np.testing.assert_allclose(pixels[:3], [[320, 240], [480, 240], [320, 120]])
        np.testing.assert_array_equal(valid, [True, True, True, False, False, False, False])

    def test_rgb_overlay_contains_target_path_and_endpoint(self):
        view, projection = self.camera_matrices()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        trajectory = np.array([[9, 20, 3], [10, 20, 3], [11, 20, 3]])
        result = overlay_goal(image, trajectory, [10, 20, 3], view, projection)
        np.testing.assert_array_equal(result[240, 240], [255, 70, 70])
        np.testing.assert_array_equal(result[240, 320], [255, 230, 0])
        np.testing.assert_array_equal(result[240, 480], [0, 230, 255])
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertFalse(image.any())


if __name__ == '__main__':
    unittest.main()
