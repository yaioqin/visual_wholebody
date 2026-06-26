import torch


class MultiAgentMessageBuilder:
    """Build explicit coordination messages between arm, base and gripper."""

    @staticmethod
    def compute_m_a2b(arm_action_chunk, arm_jacobian, eps=1.0e-6, use_svd_fallback=True):
        """Compute arm-to-base message.

        Args:
            arm_action_chunk: Tensor with shape [B, H, 6]. The first 3 dims are
                end-effector position deltas.
            arm_jacobian: Tensor with shape [B, M, N], usually [B, 6, arm_dofs].
            eps: Numerical stabilizer for determinant/SVD computations.
            use_svd_fallback: Use sqrt(prod(clamped singular values)) where the
                determinant path is non-finite or negative.

        Returns:
            Tensor with shape [B, 5] = [delta_p_bar(3), A_ee(1), eta_manip(1)].
        """
        MultiAgentMessageBuilder._validate_chunk(arm_action_chunk)
        if arm_jacobian.dim() != 3:
            raise ValueError("arm_jacobian must have shape [B, M, N]")
        if arm_action_chunk.shape[0] != arm_jacobian.shape[0]:
            raise ValueError("arm_action_chunk and arm_jacobian batch sizes must match")

        delta_p = arm_action_chunk[..., :3]
        delta_p_bar = delta_p.mean(dim=1)
        action_amplitude = torch.linalg.norm(delta_p, dim=-1).sum(dim=1, keepdim=True)
        eta_manip = MultiAgentMessageBuilder.compute_manipulability(
            arm_jacobian, eps=eps, use_svd_fallback=use_svd_fallback
        )
        return torch.cat([delta_p_bar, action_amplitude, eta_manip], dim=-1)

    @staticmethod
    def compute_m_a2g(arm_action_chunk, object_pos_wrist, sigma_p=0.25):
        """Compute arm-to-gripper message.

        Args:
            arm_action_chunk: Tensor with shape [B, H, 6].
            object_pos_wrist: Tensor with shape [B, 3], object position relative
                to wrist camera/frame.
            sigma_p: Alignment falloff length.

        Returns:
            Tensor with shape [B, 2] = [rho_align, terminal_ee_motion].
        """
        MultiAgentMessageBuilder._validate_chunk(arm_action_chunk)
        if object_pos_wrist.shape != (arm_action_chunk.shape[0], 3):
            raise ValueError("object_pos_wrist must have shape [B, 3]")

        sigma_sq = max(float(sigma_p) ** 2, 1.0e-12)
        dist_sq = torch.sum(object_pos_wrist * object_pos_wrist, dim=-1, keepdim=True)
        exponent = torch.clamp(-dist_sq / sigma_sq, min=-60.0, max=0.0)
        rho_align = torch.exp(exponent).clamp(0.0, 1.0)
        terminal_ee_motion = torch.linalg.norm(arm_action_chunk[:, -1, :3], dim=-1, keepdim=True)
        return torch.cat([rho_align, terminal_ee_motion], dim=-1)

    @staticmethod
    def compute_manipulability(arm_jacobian, eps=1.0e-6, use_svd_fallback=True):
        """Compute a batch-safe manipulability score from a Jacobian.

        The primary path is sqrt(clamp(det(J J^T + eps I), min=0)).
        If this determinant is non-finite or negative, the optional fallback uses
        sqrt(prod(clamp(singular_values(J), min=eps))) as requested for rank
        deficient or numerically unstable Jacobians.
        """
        if arm_jacobian.dim() != 3:
            raise ValueError("arm_jacobian must have shape [B, M, N]")

        batch, rows, _ = arm_jacobian.shape
        jjt = arm_jacobian @ arm_jacobian.transpose(-1, -2)
        eye = torch.eye(rows, device=arm_jacobian.device, dtype=arm_jacobian.dtype).expand(batch, rows, rows)
        det_arg = jjt + eps * eye
        det = torch.linalg.det(det_arg)
        det_is_valid = torch.isfinite(det) & (det >= 0)
        eta_det = torch.sqrt(torch.clamp(det, min=0.0)).unsqueeze(-1)

        if not use_svd_fallback:
            return torch.nan_to_num(eta_det, nan=0.0, posinf=0.0, neginf=0.0)

        singular_values = torch.linalg.svdvals(arm_jacobian)
        sv_product = torch.prod(torch.clamp(singular_values, min=eps), dim=-1)
        eta_svd = torch.sqrt(torch.clamp(sv_product, min=eps)).unsqueeze(-1)
        eta = torch.where(det_is_valid.unsqueeze(-1), eta_det, eta_svd)
        return torch.nan_to_num(eta, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def build(arm_action_chunk, arm_jacobian, object_pos_wrist, eps=1.0e-6, sigma_p=0.25):
        return {
            "m_a2b": MultiAgentMessageBuilder.compute_m_a2b(arm_action_chunk, arm_jacobian, eps=eps),
            "m_a2g": MultiAgentMessageBuilder.compute_m_a2g(arm_action_chunk, object_pos_wrist, sigma_p=sigma_p),
        }

    @staticmethod
    def _validate_chunk(arm_action_chunk):
        if arm_action_chunk.dim() != 3 or arm_action_chunk.shape[-1] != 6:
            raise ValueError("arm_action_chunk must have shape [B, H, 6]")

