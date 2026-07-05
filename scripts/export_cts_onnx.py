#!/usr/bin/env python3
"""Export CTS student policy (encoder + actor) to ONNX for deployment.

Usage:
    python scripts/export_cts_onnx.py --checkpoint logs/cts/.../model_100.pt

Output:
    logs/cts/.../cts_encoder.onnx   (240→32)
    logs/cts/.../cts_actor.onnx     (80→12)
    logs/cts/.../cts_policy.onnx    (combined: 288→12, takes history+obs at once)
"""

import argparse
import os

import torch
import torch.nn as nn

from isaaclab_rl.rsl_rl.cts_networks import CTSActor
from isaaclab_rl.rsl_rl.encoder_model import EncoderModel


def parse_args():
    parser = argparse.ArgumentParser(description="Export CTS policy to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to CTS checkpoint .pt file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: same as checkpoint)")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--obs_dim", type=int, default=48, help="Proprioceptive observation dimension")
    parser.add_argument("--history_length", type=int, default=5, help="Observation history length")
    parser.add_argument("--num_actions", type=int, default=12, help="Number of actions")
    parser.add_argument("--encoder_hidden", type=int, nargs="+", default=[512, 256], help="Encoder hidden dims")
    parser.add_argument("--actor_hidden", type=int, nargs="+", default=[512, 256, 128], help="Actor hidden dims")
    parser.add_argument("--activation", type=str, default="elu", help="Activation function")
    return parser.parse_args()


class CTSPolicyWrapper(nn.Module):
    """Wrapper combining encoder + actor for ONNX export.

    Takes both history and current obs at once:
        Input: [history(240), current_obs(48)] = 288 dims
        Output: 12-dim action
    """

    def __init__(self, encoder: EncoderModel, actor: CTSActor):
        super().__init__()
        self.encoder = encoder
        self.actor = actor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining encoder and actor.

        Args:
            x: [batch, obs_dim * (history_length + 1)]
               First 240: 5-frame history
               Last 48: current observation

        Returns:
            actions: [batch, num_actions]
        """
        history = x[:, :240]    # 5 × 48
        current_obs = x[:, 240:]  # 48

        # Encode history → latent
        z = self.encoder(history)

        # Concatenate current obs + latent → actor
        actor_input = torch.cat([current_obs, z], dim=-1)
        action = self.actor({"policy": actor_input}, stochastic_output=False)

        return action


class EncoderOnly(nn.Module):
    """Encoder-only wrapper for ONNX export."""

    def __init__(self, encoder: EncoderModel):
        super().__init__()
        self.encoder = encoder

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Encode history to latent.

        Args:
            history: [batch, 240] 5-frame history

        Returns:
            latent: [batch, 32]
        """
        return self.encoder(history)


class ActorOnly(nn.Module):
    """Actor-only wrapper for ONNX export."""

    def __init__(self, actor: CTSActor):
        super().__init__()
        self.actor = actor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Actor forward.

        Args:
            x: [batch, 80] current_obs(48) + latent(32)

        Returns:
            action: [batch, 12]
        """
        return self.actor({"policy": x}, stochastic_output=False)


def build_encoder(cfg):
    """Build proprioceptive encoder matching training config."""
    encoder = EncoderModel(
        obs=torch.zeros(1, cfg["obs_dim"]),
        obs_groups={"proprioceptive": ["proprioceptive"]},
        obs_set="proprioceptive",
        latent_dim=cfg["latent_dim"],
        hidden_dims=cfg["encoder_hidden"],
        activation=cfg["activation"],
        obs_normalization=True,
        use_history=True,
        history_length=cfg["history_length"],
    )
    return encoder


def build_actor(cfg):
    """Build actor matching training config."""
    obs_dim = cfg["obs_dim"]
    latent_dim = cfg["latent_dim"]
    actor_input_dim = obs_dim + latent_dim  # 48 + 32 = 80
    actor = CTSActor(
        input_dim=actor_input_dim,
        num_actions=cfg["num_actions"],
        hidden_dims=cfg["actor_hidden"],
        activation=cfg["activation"],
    )
    return actor


def export_onnx(model, dummy_input, output_path, input_names, output_names, dynamic_axes=None):
    """Export model to ONNX."""
    model.eval()
    model.cpu()
    dummy_input = dummy_input.cpu()

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[INFO] Exported: {output_path}")


def main():
    args = parse_args()

    # Load checkpoint
    ckpt_path = args.checkpoint
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if args.output_dir is None:
        output_dir = os.path.dirname(ckpt_path)
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    cfg = {
        "obs_dim": args.obs_dim,
        "latent_dim": args.latent_dim,
        "history_length": args.history_length,
        "num_actions": args.num_actions,
        "encoder_hidden": args.encoder_hidden,
        "actor_hidden": args.actor_hidden,
        "activation": args.activation,
    }

    # Build networks
    encoder = build_encoder(cfg)
    actor = build_actor(cfg)

    # Load weights
    encoder.load_state_dict(ckpt["proprioceptive_encoder"])
    actor.load_state_dict(ckpt["actor"])
    print(f"[INFO] Loaded checkpoint at iteration {ckpt.get('iteration', 'unknown')}")

    # ===== Export 1: Combined policy =====
    combined = CTSPolicyWrapper(encoder, actor)
    dummy_input = torch.randn(1, cfg["obs_dim"] * (cfg["history_length"] + 1))  # 288
    export_onnx(
        combined, dummy_input,
        os.path.join(output_dir, "cts_policy.onnx"),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
    )

    # ===== Export 2: Encoder only =====
    encoder_only = EncoderOnly(encoder)
    dummy_history = torch.randn(1, cfg["obs_dim"] * cfg["history_length"])  # 240
    export_onnx(
        encoder_only, dummy_history,
        os.path.join(output_dir, "cts_encoder.onnx"),
        input_names=["history"],
        output_names=["latent"],
        dynamic_axes={"history": {0: "batch"}, "latent": {0: "batch"}},
    )

    # ===== Export 3: Actor only =====
    actor_only = ActorOnly(actor)
    dummy_actor_input = torch.randn(1, cfg["obs_dim"] + cfg["latent_dim"])  # 80
    export_onnx(
        actor_only, dummy_actor_input,
        os.path.join(output_dir, "cts_actor.onnx"),
        input_names=["obs_and_latent"],
        output_names=["action"],
        dynamic_axes={"obs_and_latent": {0: "batch"}, "action": {0: "batch"}},
    )

    print(f"\n[INFO] All models exported to: {output_dir}")
    print(f"  - cts_policy.onnx  : input [B, 288] → action [B, 12]  (one-shot: history + current obs)")
    print(f"  - cts_encoder.onnx : input [B, 240] → latent [B, 32]")
    print(f"  - cts_actor.onnx   : input [B, 80]  → action [B, 12]")


if __name__ == "__main__":
    main()
