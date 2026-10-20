import torch
import torch.nn as nn
import torch.nn.functional as F


class BiMambaNAPMLoss(nn.Module):
    """
    Loss Function for BiMamba-NAPM:
    1. DeepMIL BCE Loss on coarse anomaly scores (S_det) and fused scores (S_final).
    2. Normal-Anchored Prototype Memory (NAPM) Loss:
       - Compactness Loss: Pulls normal video snippets tightly towards prototypes.
       - Push Loss: Pushes top-k abnormal video snippets away from prototypes.
       - Orthogonality Loss: Encourages diversity among prototypes.
    3. Multimodal Vision-Language Alignment Loss (optional).
    4. Temporal Smoothness & Sparsity Regularizations.
    """
    def __init__(
        self,
        loss_napm_weight: float = 0.5,
        loss_align_weight: float = 0.5,
        loss_smooth_weight: float = 1e-4,
        loss_sparse_weight: float = 5e-4,
        k_ratio: float = 16.0,
        margin: float = 0.2,
    ):
        super().__init__()
        self.loss_napm_weight = loss_napm_weight
        self.loss_align_weight = loss_align_weight
        self.loss_smooth_weight = loss_smooth_weight
        self.loss_sparse_weight = loss_sparse_weight
        self.k_ratio = k_ratio
        self.margin = margin

    def get_topk_mean(self, scores):
        """Compute mean of top-k snippet scores for each video in batch."""
        if scores.dim() == 3:
            scores = scores.squeeze(-1)
        B, T = scores.shape
        k = max(int(T / self.k_ratio) + 1, 1)
        topk_scores, _ = torch.topk(scores, k, dim=-1)
        return torch.mean(topk_scores, dim=-1)

    def compute_napm_loss(self, abn_features, nor_features, prototypes, abn_det):
        """
        Normal-Anchored Prototype Memory Loss:
        - Compactness: Normal snippets must be very close to at least one prototype.
        - Push: Top-K abnormal snippets must be distant from all normal prototypes.
        - Ortho: Prototypes must span diverse normal modes.
        """
        P = F.normalize(prototypes, p=2, dim=-1)  # (K, D)
        K, D = P.shape

        # 1. Compactness Loss on pure Normal Videos
        nor_norm = F.normalize(nor_features, p=2, dim=-1)  # (B_nor, T, D)
        B_nor, T_nor, _ = nor_norm.shape
        # Flatten all normal snippets
        nor_flat = nor_norm.view(-1, D)  # (N_nor, D)
        # Cosine distance to nearest prototype
        sim_nor = torch.matmul(nor_flat, P.transpose(0, 1))  # (N_nor, K)
        max_sim_nor, _ = torch.max(sim_nor, dim=-1)         # (N_nor,)
        loss_compact = torch.mean(1.0 - max_sim_nor)

        # 2. Push Loss on Top-K Abnormal Snippets
        abn_norm = F.normalize(abn_features, p=2, dim=-1)  # (B_abn, T, D)
        B_abn, T_abn, _ = abn_norm.shape
        k = max(int(T_abn / self.k_ratio) + 1, 1)

        # Get top-k anomaly snippet indices per video
        _, topk_idx = torch.topk(abn_det, k, dim=-1)  # (B_abn, k)
        topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        abn_topk_feat = torch.gather(abn_norm, 1, topk_idx_expanded).view(-1, D)  # (B_abn * k, D)

        sim_abn = torch.matmul(abn_topk_feat, P.transpose(0, 1))  # (B_abn * k, K)
        # Penalize if abnormal snippet has similarity > margin with any normal prototype
        loss_push = torch.mean(F.relu(sim_abn - self.margin))

        # 3. Orthogonality Loss on Prototypes
        gram = torch.matmul(P, P.transpose(0, 1))  # (K, K)
        identity = torch.eye(K, device=P.device)
        loss_ortho = torch.mean((gram - identity) ** 2)

        return loss_compact + loss_push + 0.1 * loss_ortho

    def forward(self, abn_out, nor_out, abn_class_indices=None):
        """
        Computes total loss across abnormal and normal batch outputs.
        """
        abn_det = abn_out['det_score'].squeeze(-1)      # (B, T)
        nor_det = nor_out['det_score'].squeeze(-1)      # (B, T)
        abn_final = abn_out['final_score'].squeeze(-1)  # (B, T)
        nor_final = nor_out['final_score'].squeeze(-1)  # (B, T)

        # -------------------------------------------------------------
        # 1. DeepMIL BCE Loss (Discriminative Anomaly Detection)
        # -------------------------------------------------------------
        abn_topk_det = self.get_topk_mean(abn_det)
        nor_topk_det = self.get_topk_mean(nor_det)

        loss_bce_abn = -torch.mean(torch.log(abn_topk_det.clamp(min=1e-6, max=1.0)))
        loss_bce_nor = -torch.mean(torch.log((1.0 - nor_topk_det).clamp(min=1e-6, max=1.0)))
        loss_bce_det = 0.5 * (loss_bce_abn + loss_bce_nor)

        abn_topk_final = self.get_topk_mean(abn_final)
        nor_topk_final = self.get_topk_mean(nor_final)
        loss_bce_final = 0.5 * (
            -torch.mean(torch.log(abn_topk_final.clamp(min=1e-6, max=1.0)))
            - torch.mean(torch.log((1.0 - nor_topk_final).clamp(min=1e-6, max=1.0)))
        )
        loss_mil = loss_bce_det + loss_bce_final

        # -------------------------------------------------------------
        # 2. Normal-Anchored Prototype Memory (NAPM) Loss
        # -------------------------------------------------------------
        prototypes = abn_out['prototypes']
        loss_napm = self.compute_napm_loss(
            abn_features=abn_out['features'],
            nor_features=nor_out['features'],
            prototypes=prototypes,
            abn_det=abn_det.detach()
        )

        # -------------------------------------------------------------
        # 3. Vision-Language Alignment Loss (Optional)
        # -------------------------------------------------------------
        loss_align = torch.tensor(0.0, device=abn_det.device)
        abn_probs = abn_out.get('class_probs')
        nor_probs = nor_out.get('class_probs')

        if abn_probs is not None and nor_probs is not None and abn_class_indices is not None:
            B, T, C = abn_probs.shape
            k = max(int(T / self.k_ratio) + 1, 1)

            # Normal videos: All snippets must align to Class 0 (Normal)
            loss_align_nor = -torch.mean(torch.log(nor_probs[:, :, 0].clamp(min=1e-6, max=1.0)))

            # Abnormal videos: Top-K snippets align to specific crime class
            event_loss_list = []
            for b in range(B):
                target_cls = int(abn_class_indices[b].item())
                _, topk_idx = torch.topk(abn_det[b], k)
                if 0 < target_cls < C:
                    event_prob = abn_probs[b, topk_idx, target_cls]
                else:
                    event_prob = abn_probs[b, topk_idx, 1:].sum(dim=-1)
                event_loss_list.append(-torch.mean(torch.log(event_prob.clamp(min=1e-6, max=1.0))))

            loss_align_abn = torch.stack(event_loss_list).mean()
            loss_align = 0.5 * (loss_align_nor + loss_align_abn)

        # -------------------------------------------------------------
        # 4. Temporal Smoothness & Sparsity
        # -------------------------------------------------------------
        diff_abn = abn_det[:, 1:] - abn_det[:, :-1]
        loss_smooth = torch.mean(diff_abn ** 2)
        loss_sparse = torch.mean(abn_det)

        # Total Loss
        total_loss = (
            loss_mil
            + self.loss_napm_weight * loss_napm
            + self.loss_align_weight * loss_align
            + self.loss_smooth_weight * loss_smooth
            + self.loss_sparse_weight * loss_sparse
        )

        return {
            'total_loss': total_loss,
            'loss_mil': loss_mil,
            'loss_napm': loss_napm,
            'loss_align': loss_align,
            'loss_smooth': loss_smooth,
            'loss_sparse': loss_sparse,
        }
