import os
import sys
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import UCFDataset
from model import BiMambaNAPM
from losses import BiMambaNAPMLoss
from options import get_parser
from ucf_test import evaluate_ucf
from text_prompts import load_or_encode_text_features, get_class_index
from tools import get_batch_mask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_ucf():
    parser = get_parser(dataset='ucf')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.set_device(0)
        # Establish primary CUDA cuBLAS context to eliminate cuBLAS warning
        _ = torch.zeros(1, device='cuda')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"\n=======================================================")
    print(f"  BiMamba-NAPM: Training from Scratch on UCF-Crime")
    print(f"  Target: UCF-Crime AUC >= 90.0% | Scratch (No Warm-Start)")
    print(f"  SSM Type: {args.ssm_type} | Layers: {args.num_layers} | Prototypes: {args.num_prototypes}")
    print(f"  Device: {device} | Batch Size: {args.batch_size} | Max Epochs: {args.max_epoch}")
    print(f"=======================================================\n", flush=True)

    # 1. Datasets
    train_abn_dataset = UCFDataset(args.visual_length, args.train_list, test_mode=False, normal=False)
    train_nor_dataset = UCFDataset(args.visual_length, args.train_list, test_mode=False, normal=True)

    train_abn_loader = DataLoader(
        train_abn_dataset, batch_size=args.batch_size // 2, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    train_nor_loader = DataLoader(
        train_nor_dataset, batch_size=args.batch_size // 2, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    test_dataset = UCFDataset(args.visual_length, args.test_list, test_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    # 2. Text Prompts
    text_features = None
    if args.use_text_prompt:
        text_features = load_or_encode_text_features(device=device, dataset='ucf')

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

    # DataParallel for dual GPUs (e.g. 2x T4 on Kaggle)
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

    best_auc = 0.0
    best_ap = 0.0
    total_steps = len(train_abn_loader)

    # 5. Training Loop
    for epoch in range(1, args.max_epoch + 1):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        running_mil = 0.0
        running_napm = 0.0

        nor_iter = iter(train_nor_loader)
        print(f"--- Epoch [{epoch:02d}/{args.max_epoch:02d}] Started ---", flush=True)

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

            # Class indices for semantic loss
            abn_class_indices = torch.tensor(
                [get_class_index(lbl, 'ucf') for lbl in abn_labels],
                dtype=torch.long, device=device
            )

            optimizer.zero_grad()

            # Forward passes
            abn_out = model(abn_visual, padding_mask=abn_mask, text_features=text_features)
            nor_out = model(nor_visual, padding_mask=nor_mask, text_features=text_features)

            # Compute Normal-Anchored loss
            loss_dict = criterion(abn_out, nor_out, abn_class_indices)
            loss = loss_dict['total_loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            running_mil += loss_dict['loss_mil'].item()
            running_napm += loss_dict['loss_napm'].item()

            if (step + 1) % 5 == 0 or (step + 1) == total_steps:
                print(f"  Step [{step+1:02d}/{total_steps:02d}] | "
                      f"Loss: {loss.item():.4f} (MIL: {loss_dict['loss_mil'].item():.4f}, NAPM: {loss_dict['loss_napm'].item():.4f})", flush=True)

        scheduler.step()
        epoch_time = time.time() - start_time
        avg_loss = running_loss / total_steps
        avg_mil = running_mil / total_steps
        avg_napm = running_napm / total_steps

        print(f"Epoch [{epoch:02d}/{args.max_epoch:02d}] Summary ({epoch_time:.1f}s) | "
              f"Avg Loss: {avg_loss:.4f} (MIL: {avg_mil:.4f}, NAPM: {avg_napm:.4f}) | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)

        # Validation
        print(f"  Evaluating on test set ({len(test_loader)} videos)...", flush=True)
        eval_model = model.module if hasattr(model, 'module') else model
        auc, ap = evaluate_ucf(eval_model, test_loader, args.visual_length, gt, gtsegments, gtlabels, device, text_features)

        if auc > best_auc:
            best_auc = auc
            best_ap = ap
            save_path = os.path.join(args.save_dir, "bimamba_napm_ucf_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': eval_model.state_dict(),
                'best_auc': best_auc,
                'best_ap': best_ap,
                'args': vars(args)
            }, save_path)
            print(f"  🏆 NEW BEST SAVED: AUC = {best_auc:.2f}% | AP = {best_ap:.2f}% -> {save_path}\n", flush=True)

    print(f"\n=======================================================")
    print(f"  Training Complete! Peak Performance:")
    print(f"  ⭐ Peak ROC-AUC: {best_auc:.2f}% (Target: >= 90.0%)")
    print(f"  ⭐ Corresponding AP: {best_ap:.2f}%")
    print(f"=======================================================\n", flush=True)


if __name__ == '__main__':
    train_ucf()
