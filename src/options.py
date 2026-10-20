import argparse


def get_parser(dataset='ucf'):
    parser = argparse.ArgumentParser(description=f'BiMamba-NAPM Training & Evaluation ({dataset.upper()})')

    # General & Hardware
    parser.add_argument('--dataset', type=str, default=dataset, choices=['ucf', 'xd'])
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--num_workers', type=int, default=4)

    # Model Hyperparameters
    parser.add_argument('--in_dim', type=int, default=512, help='Input CLIP feature dimension')
    parser.add_argument('--d_model', type=int, default=512, help='State space model dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of stacked BiMamba blocks')
    parser.add_argument('--d_state', type=int, default=16, help='SSM state dimension')
    parser.add_argument('--ssm_type', type=str, default='s4d', choices=['s4d', 'pytorch', 'official'])
    parser.add_argument('--num_prototypes', type=int, default=16, help='Number of normal-anchored prototypes')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--gate_init_bias', type=float, default=0.0)

    # Optimization
    parser.add_argument('--lr', type=float, default=5e-5 if dataset == 'ucf' else 1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=15 if dataset == 'ucf' else 20)

    # Loss Weights
    parser.add_argument('--loss_napm_weight', type=float, default=0.5, help='Weight for Normal-Anchored Prototype Memory')
    parser.add_argument('--loss_align_weight', type=float, default=0.5, help='Weight for Multimodal Vision-Language Alignment')
    parser.add_argument('--loss_smooth_weight', type=float, default=1e-4)
    parser.add_argument('--loss_sparse_weight', type=float, default=5e-4)
    parser.add_argument('--k_ratio', type=float, default=16.0)
    parser.add_argument('--margin', type=float, default=0.2)

    # Dataset & Paths
    if dataset == 'ucf':
        parser.add_argument('--visual_length', type=int, default=256)
        parser.add_argument('--train_list', type=str, default='list/ucf_CLIP_rgb.csv')
        parser.add_argument('--test_list', type=str, default='list/ucf_CLIP_rgbtest.csv')
        parser.add_argument('--gt_path', type=str, default='list/gt_ucf.npy')
        parser.add_argument('--gt_segment_path', type=str, default='list/gt_segment_ucf.npy')
        parser.add_argument('--gt_label_path', type=str, default='list/gt_label_ucf.npy')
        parser.add_argument('--save_dir', type=str, default='models/ucf')
    else:
        parser.add_argument('--visual_length', type=int, default=256)
        parser.add_argument('--train_list', type=str, default='list/xd_CLIP_rgb.csv')
        parser.add_argument('--test_list', type=str, default='list/xd_CLIP_rgbtest.csv')
        parser.add_argument('--gt_path', type=str, default='list/gt.npy')
        parser.add_argument('--gt_segment_path', type=str, default='list/gt_segment.npy')
        parser.add_argument('--gt_label_path', type=str, default='list/gt_label.npy')
        parser.add_argument('--save_dir', type=str, default='models/xd')

    # Pretrained & Test
    parser.add_argument('--model_path', type=str, default=None, help='Path to checkpoint for evaluation')
    parser.add_argument('--use_text_prompt', action='store_true', default=True)

    return parser
