import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from dataset import XDDataset
from tools import get_batch_mask
from model import BiMambaNAPM
from options import get_parser
from text_prompts import load_or_encode_text_features


def evaluate_xd(model, test_loader, maxlen, gt, gtsegments, gtlabels, device, text_features=None):
    """
    Evaluates BiMamba-NAPM on XD-Violence test set.
    Computes frame-level Average Precision (AP) and ROC-AUC.
    """
    model.eval()

    all_scores_final = []
    all_scores_det = []
    all_scores_rec = []

    with torch.no_grad():
        for i, item in enumerate(test_loader):
            visual = item[0].squeeze(0)
            length = int(item[2].item()) if torch.is_tensor(item[2]) else int(item[2])
            len_cur = length

            if visual.dim() == 2:
                visual = visual.unsqueeze(0)
            visual = visual.to(device)

            lengths = torch.tensor([min(length, maxlen)], dtype=torch.long, device=device)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)

            out = model(visual[:, :maxlen, :], padding_mask=padding_mask, text_features=text_features)

            s_final = out['final_score'][0, :len_cur].squeeze(-1).cpu().numpy()
            s_det = out['det_score'][0, :len_cur].squeeze(-1).cpu().numpy()
            s_rec = out['rec_score'][0, :len_cur].squeeze(-1).cpu().numpy()

            all_scores_final.append(s_final)
            all_scores_det.append(s_det)
            all_scores_rec.append(s_rec)

            if (i + 1) % 100 == 0 or (i + 1) == len(test_loader):
                print(f"    Testing [{i+1:03d}/{len(test_loader):03d}] videos...", flush=True)

    all_scores_final = np.concatenate(all_scores_final, axis=0)
    all_scores_det = np.concatenate(all_scores_det, axis=0)
    all_scores_rec = np.concatenate(all_scores_rec, axis=0)

    # Frame-level expansion (16 frames per snippet)
    preds_final_frames = np.repeat(all_scores_final, 16)
    preds_det_frames = np.repeat(all_scores_det, 16)
    preds_rec_frames = np.repeat(all_scores_rec, 16)

    min_len = min(len(gt), len(preds_final_frames))
    gt_eval = gt[:min_len]

    ap_final = average_precision_score(gt_eval, preds_final_frames[:min_len]) * 100.0
    auc_final = roc_auc_score(gt_eval, preds_final_frames[:min_len]) * 100.0

    ap_det = average_precision_score(gt_eval, preds_det_frames[:min_len]) * 100.0
    auc_det = roc_auc_score(gt_eval, preds_det_frames[:min_len]) * 100.0

    ap_rec = average_precision_score(gt_eval, preds_rec_frames[:min_len]) * 100.0
    auc_rec = roc_auc_score(gt_eval, preds_rec_frames[:min_len]) * 100.0

    print(f"\n[XD-Violence Test Results]")
    print(f"  ⭐ Final Fused   : AP = {ap_final:.2f}% | AUC = {auc_final:.2f}%")
    print(f"  🔹 BiMamba Det   : AP = {ap_det:.2f}% | AUC = {auc_det:.2f}%")
    print(f"  🔸 Normal Memory : AP = {ap_rec:.2f}% | AUC = {auc_rec:.2f}%")

    return ap_final, auc_final


if __name__ == '__main__':
    parser = get_parser(dataset='xd')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    test_dataset = XDDataset(args.visual_length, args.test_list, test_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    text_features = None
    if args.use_text_prompt:
        text_features = load_or_encode_text_features(device=device, dataset='xd')

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

    evaluate_xd(model, test_loader, args.visual_length, gt, gtsegments, gtlabels, device, text_features)
