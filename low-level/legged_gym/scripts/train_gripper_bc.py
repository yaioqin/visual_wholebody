from pathlib import Path
import argparse
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


LOW_LEVEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LOW_LEVEL_ROOT))

from legged_gym.controllers.gripper_controller import GripperPolicyMLP


def load_dataset(path, num_samples):
    if path is None:
        grasp_intent = torch.rand(num_samples, 1)
        rho_align = torch.rand(num_samples, 1)
        terminal_motion = torch.rand(num_samples, 1) * 0.05
        obs = torch.cat([grasp_intent, rho_align, terminal_motion], dim=-1)
    else:
        data = torch.load(path, map_location="cpu")
        if "obs" in data:
            obs = data["obs"].float()
        else:
            obs = torch.cat(
                [
                    data["grasp_intent"].float(),
                    data["rho_align"].float(),
                    data["terminal_ee_motion"].float(),
                ],
                dim=-1,
            )

    labels = (
        (obs[:, 0:1] > 0.5)
        & (obs[:, 1:2] > 0.75)
        & (obs[:, 2:3] < 0.02)
    ).float()
    return TensorDataset(obs, labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="gripper_policy_bc.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.num_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = GripperPolicyMLP(obs_dim=3, hidden_dim=64, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        total_loss = 0.0
        for obs, label in loader:
            logit = model(obs)
            loss = criterion(logit, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * obs.shape[0]
        print(f"epoch={epoch + 1} loss={total_loss / len(dataset):.6f}")

    torch.save({"model_state_dict": model.state_dict()}, args.save_path)
    print(f"saved gripper BC checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()

