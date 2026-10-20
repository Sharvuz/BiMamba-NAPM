import os
import sys
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import XDDataset
from model import BiMambaNAPM
from losses import BiMambaNAPMLoss
from options import get_parser
from xd_test import evaluate_xd
from text_prompts import load_or_encode_text_features, get_class_index
from tools import get_batch_mask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_xd():
    parser = get_parser(dataset='xd')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    print(f"\n=======================================================")
    print(f"  BiMamba-NAPM: Training from Scratch on XD-Violence")
    print(f"  Target: XD-Violence AP > 87.0% | Scratch (No Warm-Start)")
    print(f"  SSM Type: {args.ssm_type} | Layers: {args.num_layers} | Prototypes: {args.num_prototypes}")
    print(f"  Device: {device} | Batch Size: {args.batch_size} | Max Epochs: {args.max_epoch}")
    print(f"=======================================================\n", flush=True)

    # 1. Datasets
    train_abn_dataset = XDDataset(args.visual_length, args.train_list, test_mode=False, normal=False)
    train_nor_dataset = XDDataset(args.visual_length, args.train_list, test_mode=False, normal=True)

    train_abn_loader = DataLoader(
        train_abn_dataset, batch_size=args.batch_size // 2, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    train_nor_loader = DataLoader(
        train_nor_dataset, batch_size=args.batch_size // 2, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    test_dataset = XDDataset(args.visual_length, args.test_list, test_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    # 2. Text Prompts
    text_features = None
    if args.use_text_prompt:
        text_features = load_or_encode_text_features(device=device, dataset='xd')

    # 3. Model
    model = BiMambaNAPM(
        in_dim=args.in_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        d_state=args.d_state,
        ssm_type=args.ssm_type,
        num_prototypes=args.num_prototypes,
        dropout=args.dropout,
        gate_init_bias=args.gate_init_bias
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}\n", flush=True)

    # DataParallel for dual GPUs
    raw_model = model
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        print(f"🔥 Multi-GPU detected: Running on {torch.cuda.device_count()} GPUs (DataParallel)!\n", flush=True)
        model = torch.nn.DataParallel(model)

    # 4. Optimization & Loss
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=1e-6)

    criterion = BiMambaNAPMLoss(
        loss_napm_weight=args.loss_napm_weight,
        loss_align_weight=args.loss_align_weight,
        loss_smooth_weight=args.loss_smooth_weight,
        loss_sparse_weight=args.loss_sparse_weight,
        k_ratio=args.k_ratio,
        margin=args.margin
    )

    best_ap = 0.0
    best_auc = 0.0

    # 5. Training Loop
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        running_mil = 0.0
        running_napm = 0.0

        nor_iter = iter(train_nor_loader)

        for step, (abn_visual, abn_labels, abn_lengths) in enumerate(train_abn_loader):
            try:
                nor_visual, _, nor_lengths = next(nor_iter)
            except StopIteration:
                nor_iter = iter(train_nor_loader)
                nor_visual, _, nor_lengths = next(nor_iter)

            abn_visual = abn_visual.to(device)
            nor_visual = nor_visual.to(device)

            abn_mask = get_batch_mask(abn_lengths, args.visual_length).to(device)
            nor_mask = get_batch_mask(nor_lengths, args.visual_length).to(device)

            abn_class_indices = torch.tensor(
                [get_class_index(lbl, 'xd') for lbl in abn_labels],
                dtype=torch.long, device=device
            )

            optimizer.zero_grad()

            abn_out = model(abn_visual, padding_mask=abn_mask, text_features=text_features)
            nor_out = model(nor_visual, padding_mask=nor_mask, text_features=text_features)

            loss_dict = criterion(abn_out, nor_out, abn_class_indices)
            loss = loss_dict['total_loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            running_mil += loss_dict['loss_mil'].item()
            running_napm += loss_dict['loss_napm'].item()

        scheduler.step()
        epoch_time = time.time() - start_time
        num_steps = len(train_abn_loader)
        avg_loss = running_loss / num_steps
        avg_mil = running_mil / num_steps
        avg_napm = running_napm / num_steps

        print(f"Epoch [{epoch:02d}/{args.max_epoch:02d}] ({epoch_time:.1f}s) | "
              f"Loss: {avg_loss:.4f} (MIL: {avg_mil:.4f}, NAPM: {avg_napm:.4f}) | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)

        # Validation
        eval_model = model.module if hasattr(model, 'module') else model
        ap, auc = evaluate_xd(eval_model, test_loader, args.visual_length, gt, gtsegments, gtlabels, device, text_features)

        if ap > best_ap:
            best_ap = ap
            best_auc = auc
            save_path = os.path.join(args.save_dir, "bimamba_napm_xd_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': eval_model.state_dict(),
                'best_ap': best_ap,
                'best_auc': best_auc,
                'args': vars(args)
            }, save_path)
            print(f"  🏆 NEW BEST SAVED: AP = {best_ap:.2f}% | AUC = {best_auc:.2f}% -> {save_path}\n", flush=True)

    print(f"\n=======================================================")
    print(f"  Training Complete! Peak Performance:")
    print(f"  ⭐ Peak Average Precision (AP): {best_ap:.2f}% (Target: > 87.0%)")
    print(f"  ⭐ Corresponding ROC-AUC: {best_auc:.2f}%")
    print(f"=======================================================\n", flush=True)


if __name__ == '__main__':
    train_xd()
