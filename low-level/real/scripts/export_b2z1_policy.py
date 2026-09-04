#!/usr/bin/env python3
"""Export a B2-Z1 checkpoint as a history-encoding TorchScript policy.

This exporter intentionally has no Isaac Gym or rsl_rl dependency, so it can
run on the Ubuntu 20.04 deployment computer as long as PyTorch is installed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn


NUM_PROPRIO = 71
HISTORY_LEN = 10
HISTORY_LATENT = 20
POLICY_INPUT_DIM = NUM_PROPRIO * (HISTORY_LEN + 1)


class HistoryEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(NUM_PROPRIO, 30), nn.ELU())
        self.conv_layers = nn.Sequential(
            nn.Conv1d(30, 20, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv1d(20, 10, kernel_size=2, stride=1),
            nn.ELU(),
            nn.Flatten(),
        )
        self.linear_output = nn.Sequential(nn.Linear(30, HISTORY_LATENT), nn.ELU())

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        batch = history.shape[0]
        projection = self.encoder(history.reshape(batch * 10, 71))
        projection = projection.reshape(batch, 10, 30).permute(0, 2, 1)
        return self.linear_output(self.conv_layers(projection))


class DeploymentPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.history_encoder = HistoryEncoder()
        self.actor_backbone = nn.Sequential(nn.Linear(NUM_PROPRIO + HISTORY_LATENT, 128), nn.ELU())
        self.actor_leg_control_head = nn.Sequential(
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 12)
        )
        self.actor_arm_control_head = nn.Sequential(
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 6)
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        current = observation[:, :71]
        history = observation[:, 71:].reshape(-1, 10, 71)
        latent = self.history_encoder(history)
        backbone = self.actor_backbone(torch.cat((current, latent), dim=1))
        return torch.cat(
            (self.actor_leg_control_head(backbone), self.actor_arm_control_head(backbone)), dim=1
        )


def _load_submodule(module: nn.Module, state, prefix: str) -> None:
    selected = {name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)}
    missing, unexpected = module.load_state_dict(selected, strict=False)
    if missing or unexpected:
        raise ValueError(f"checkpoint mismatch for {prefix}: missing={missing}, unexpected={unexpected}")


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(checkpoint_path: str, output_path: str) -> None:
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(checkpoint_file)
    try:
        checkpoint = torch.load(str(checkpoint_file), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_file), map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint does not contain model_state_dict")
    state = checkpoint["model_state_dict"]

    history_weight = state.get("actor.history_encoder.encoder.0.weight")
    if history_weight is None or tuple(history_weight.shape) != (30, NUM_PROPRIO):
        shape = None if history_weight is None else tuple(history_weight.shape)
        raise ValueError(
            f"checkpoint history input is {shape}; expected (30, {NUM_PROPRIO}). "
            "Was it trained with --task b2z1 --observe_gait_commands?"
        )

    model = DeploymentPolicy()
    _load_submodule(model.history_encoder, state, "actor.history_encoder.")
    _load_submodule(model.actor_backbone, state, "actor.actor_backbone.")
    _load_submodule(model.actor_leg_control_head, state, "actor.actor_leg_control_head.")
    _load_submodule(model.actor_arm_control_head, state, "actor.actor_arm_control_head.")
    model.eval()

    with torch.inference_mode():
        smoke_output = model(torch.zeros(2, POLICY_INPUT_DIM))
    if tuple(smoke_output.shape) != (2, 18) or not torch.isfinite(smoke_output).all():
        raise RuntimeError("exported network failed the finite-value smoke test")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model)
    scripted.save(str(output_file))
    metadata = {
        "task": "b2z1",
        "observe_gait_commands": True,
        "num_proprio": NUM_PROPRIO,
        "history_len": HISTORY_LEN,
        "input_dim": POLICY_INPUT_DIM,
        "output_dim": 18,
        "checkpoint": str(checkpoint_file),
        "checkpoint_sha256": _checkpoint_hash(checkpoint_file),
        "checkpoint_iteration": int(checkpoint.get("iter", -1)),
    }
    metadata_file = output_file.with_suffix(output_file.suffix + ".json")
    metadata_file.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Exported policy: {output_file}")
    print(f"Metadata: {metadata_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Training model_*.pt checkpoint")
    parser.add_argument("--output", required=True, help="Destination TorchScript .pt path")
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
