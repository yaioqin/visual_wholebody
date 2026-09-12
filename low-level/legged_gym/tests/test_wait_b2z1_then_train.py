"""Exercise the queue in isolated tmux servers using a dummy training program."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time
import unittest


WAITER = Path(__file__).resolve().parents[1] / "scripts" / "wait_b2z1_then_train.sh"
EXPECTED_ARGS = [
    "--headless", "--exptid", "b2_z1_change_ee_orn_range",
    "--proj_name", "b2z1-low", "--task", "b2z1",
    "--resumeid", "b2_z1_change_ee_pos_range", "--checkpoint", "38000",
    "--max_iterations", "45000", "--sim_device", "cuda:0",
    "--rl_device", "cuda:0", "--observe_gait_commands",
]


@unittest.skipUnless(shutil.which("tmux"), "tmux is required")
class TrainingQueueTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="b2z1 queue test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scripts = self.root / "legged_gym" / "scripts"
        self.scripts.mkdir(parents=True)
        self.waiter = self.scripts / WAITER.name
        shutil.copyfile(WAITER, self.waiter)
        self.checkpoint = self.root / "logs/b2z1-low/b2_z1_change_ee_pos_range/model_38000.pt"
        self.checkpoint.parent.mkdir(parents=True)
        self.socket = self.root / "tmux.sock"
        (self.scripts / "train.py").write_text(
            "import json, os, pathlib, sys, time\n"
            "if sys.argv[1:] == ['--fake-current']:\n"
            "    pathlib.Path('current_started').touch()\n"
            "    time.sleep(3)\n"
            "    pathlib.Path('current_finished').touch()\n"
            "else:\n"
            "    with open('launches.jsonl', 'a') as stream:\n"
            "        stream.write(json.dumps({\n"
            "            'args': sys.argv[1:], 'cwd': os.getcwd(),\n"
            "            'env': os.environ.get('QUEUE_TEST_ENV'),\n"
            "            'current_finished': pathlib.Path('current_finished').exists()\n"
            "        }) + '\\n')\n"
        )
        self.tmux("new-session", "-d", "-s", "b2z1", "-c", str(self.scripts),
                  "bash --noprofile --norc")
        self.addCleanup(self.stop_server)
        self.pane = self.tmux("display-message", "-p", "-t", "b2z1", "#{pane_id}").strip()
        server_pid = self.tmux("display-message", "-p", "-t", self.pane, "#{pid}").strip()
        self.waiter_env = dict(os.environ, TMUX=f"{self.socket},{server_pid},0",
                               B2Z1_QUEUE_POLL_SECONDS="1", TMPDIR=str(self.root))
        self.waiter_env.pop("TMUX_PANE", None)

    def tmux(self, *args):
        return subprocess.check_output(
            ["tmux", "-S", str(self.socket), *args], text=True, stderr=subprocess.STDOUT
        )

    def stop_server(self):
        subprocess.run(["tmux", "-S", str(self.socket), "kill-server"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def wait_for_file(self, path):
        deadline = time.monotonic() + 10
        while not path.exists():
            if time.monotonic() >= deadline:
                self.fail(f"Timed out waiting for {path}")
            time.sleep(0.05)

    def start_current_training(self):
        command = "export QUEUE_TEST_ENV=preserved; " + shlex.join(
            [shutil.which("python"), "train.py", "--fake-current"]
        )
        self.tmux("send-keys", "-t", self.pane, "-l", command)
        self.tmux("send-keys", "-t", self.pane, "Enter")
        self.wait_for_file(self.scripts / "current_started")

    def start_waiter(self):
        process = subprocess.Popen(
            ["bash", str(self.waiter)], env=self.waiter_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        def cleanup():
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=5)
        self.addCleanup(cleanup)
        return process

    def test_waits_then_runs_exact_command_in_original_shell(self):
        self.checkpoint.write_bytes(b"dummy checkpoint")
        self.start_current_training()
        waiter = self.start_waiter()
        output, _ = waiter.communicate(timeout=15)
        self.assertEqual(waiter.returncode, 0, output)
        launch_file = self.scripts / "launches.jsonl"
        self.wait_for_file(launch_file)
        launches = [json.loads(line) for line in launch_file.read_text().splitlines()]
        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0]["args"], EXPECTED_ARGS)
        self.assertEqual(launches[0]["cwd"], str(self.scripts))
        self.assertEqual(launches[0]["env"], "preserved")
        self.assertTrue(launches[0]["current_finished"])

    def test_missing_checkpoint_does_not_start_training(self):
        waiter = self.start_waiter()
        output, _ = waiter.communicate(timeout=10)
        self.assertNotEqual(waiter.returncode, 0)
        self.assertIn("Checkpoint missing or empty", output)
        self.assertFalse((self.scripts / "launches.jsonl").exists())

    def test_duplicate_waiter_is_rejected(self):
        self.checkpoint.write_bytes(b"dummy checkpoint")
        self.start_current_training()
        first = self.start_waiter()
        self.assertIn("Waiting for", first.stdout.readline())
        second = self.start_waiter()
        output, _ = second.communicate(timeout=10)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("A waiter is already running", output)
        first_output, _ = first.communicate(timeout=15)
        self.assertEqual(first.returncode, 0, first_output)


if __name__ == "__main__":
    unittest.main()
