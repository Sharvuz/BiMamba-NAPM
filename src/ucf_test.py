import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from dataset import UCFDataset
from tools import get_batch_mask
from model import BiMambaNAPM
from options import get_parser
from text_prompts import load_or_encode_text_features


def evaluate_ucf(model, test_loader, maxlen, gt, gtsegments, gtlabels, device, text_features=None):
    """
    Evaluates BiMamba-NAPM on UCF-Crime test set.
    Computes frame-level ROC-AUC and Average Precision (AP).
    """
    model.eval()

    all_scores_final = []
    all_scores_det = []
    all_scores_rec = []
    all_scores_sem = []

    with torch.no_grad():
        for i, item in enumerate(test_loader):
            visual = item[0].squeeze(0)  # (split_num, maxlen, 512) or (maxlen, 512)
            length = int(item[2].item()) if torch.is_tensor(item[2]) else int(item[2])
            len_cur = length

            if visual.dim() == 2:
                visual = visual.unsqueeze(0)  # (1, maxlen, 512)
            visual = visual.to(device)
            split_num = visual.shape[0]

            # Construct chunk lengths for all splits
            rem_len = len_cur
            lengths = torch.zeros(split_num, dtype=torch.long, device=device)
            for j in range(split_num):
                if rem_len > maxlen:
                    lengths[j] = maxlen
                    rem_len -= maxlen
                else:
                    lengths[j] = rem_len

            padding_mask = get_batch_mask(lengths, maxlen).to(device)

            out = model(visual, padding_mask=padding_mask, text_features=text_features)

            # Flatten across all chunks and slice to true video length len_cur
            s_final = out['final_score'].reshape(-1)[:len_cur].cpu().numpy()
            s_det = out['det_score'].reshape(-1)[:len_cur].cpu().numpy()
            s_rec = out['rec_score'].reshape(-1)[:len_cur].cpu().numpy()

            all_scores_final.append(s_final)
            all_scores_det.append(s_det)
            all_scores_rec.append(s_rec)

            if out['sem_score'] is not None:
                s_sem = out['sem_score'].reshape(-1)[:len_cur].cpu().numpy()
                all_scores_sem.append(s_sem)

            if (i + 1) % 50 == 0 or (i + 1) == len(test_loader):
                print(f"    Testing [{i+1:03d}/{len(test_loader):03d}] videos...", flush=True)

    all_scores_final = np.concatenate(all_scores_final, axis=0)
    all_scores_det = np.concatenate(all_scores_det, axis=0)
    all_scores_rec = np.concatenate(all_scores_rec, axis=0)

    # Frame-level expansion: each snippet covers 16 frames
    preds_final_frames = np.repeat(all_scores_final, 16)
    preds_det_frames = np.repeat(all_scores_det, 16)
    preds_rec_frames = np.repeat(all_scores_rec, 16)

    # Ensure equal length with ground truth
    min_len = min(len(gt), len(preds_final_frames))
    gt_eval = gt[:min_len]

    auc_final = roc_auc_score(gt_eval, preds_final_frames[:min_len]) * 100.0
    ap_final = average_precision_score(gt_eval, preds_final_frames[:min_len]) * 100.0

    auc_det = roc_auc_score(gt_eval, preds_det_frames[:min_len]) * 100.0
    ap_det = average_precision_score(gt_eval, preds_det_frames[:min_len]) * 100.0

    auc_rec = roc_auc_score(gt_eval, preds_rec_frames[:min_len]) * 100.0
    ap_rec = average_precision_score(gt_eval, preds_rec_frames[:min_len]) * 100.0

    print(f"\n[UCF-Crime Test Results]")
    print(f"  ⭐ Final Fused   : AUC = {auc_final:.2f}% | AP = {ap_final:.2f}%")
    print(f"  🔹 BiMamba Det   : AUC = {auc_det:.2f}% | AP = {ap_det:.2f}%")
    print(f"  🔸 Normal Memory : AUC = {auc_rec:.2f}% | AP = {ap_rec:.2f}%")

    return auc_final, ap_final


if __name__ == '__main__':
    parser = get_parser(dataset='ucf')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, test_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    text_features = None
    if args.use_text_prompt:
        text_features = load_or_encode_text_features(device=device, dataset='ucf')

    model = BiMambaNAPM(
        in_dim=args.in_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        d_state=args.d_state,
        ssm_type=args.ssm_type,
        num_prototypes=args.num_prototypes,
        dropout=args.dropout
    ).to(device)

    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading checkpoint from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
    else:
        print("[Warning] No model_path provided or file not found. Evaluating with random initialization.")

    evaluate_ucf(model, test_loader, args.visual_length, gt, gtsegments, gtlabels, device, text_features)
