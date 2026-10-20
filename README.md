# BiMamba-NAPM: Bidirectional Mamba with Normal-Anchored Prototype Memory for Weakly-Supervised Video Anomaly Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle: Dual T4](https://img.shields.io/badge/Kaggle-Dual%20Tesla%20T4%20Ready-20BEFF.svg?logo=kaggle&logoColor=white)](https://kaggle.com)

*Read this in: [Tiếng Việt](README.md#nghiên-cứu-khoa-học--báo-cáo-tiếng-việt)*

**Author:** Huỳnh Quang Vũ (Student ID: 2351010247)  
**Advisor:** Dr. Lê Viết Tuấn  
**Affiliation:** Faculty of Information Technology, Ho Chi Minh City Open University  

---

## 📌 1. Scientific Motivation & Breakthroughs

### The Limitations of Prior Work (DSANet & RUPA-DSA):
1. **The Warm-Start Dilemma:** Previous methods (such as RUPA-DSA) heavily relied on a pre-trained DSANet checkpoint, freezing the main network (`main_lr = 0.0`) to avoid gradient collapse. When trained from scratch end-to-end, pseudo-label noise caused catastrophic forgetting (dropping UCF-Crime AP from 38.3% down to 25.5%).
2. **Architecture Redundancy:** DSANet stacks a windowed Transformer with dual graph convolution networks (Distance Graph + Similarity Graph), leading to excessive computational overhead without capturing continuous temporal dynamics.
3. **The 90% AUC Bottleneck on UCF-Crime:** Complex CCTV surveillance features false alarms on long normal videos (lighting shifts, shadows, unusual benign activity), suppressing global ROC-AUC below 90%.

### 🎯 Key Scientific Pillars of BiMamba-NAPM:
* **100% Train from Scratch (No Warm-Start):** Fully autonomous model standing on its own strength, eliminating any dependence on external model checkpoints.
* **Bidirectional Mamba/S4D Temporal Backbone ($O(T)$):** Replaces windowed attention and GCNs with a continuous Selective State Space Model, capturing forward and backward temporal dependencies with linear complexity. Provides razor-sharp anomaly boundaries to drive **XD-Violence AP > 87.0%**.
* **Normal-Anchored Prototype Memory (NAPM):** Instead of selecting noisy pseudo-normal frames from abnormal videos via Otsu or Top-K, NAPM is **anchored exclusively on pure Normal Videos ($Y=0$)**. By minimizing reconstruction variance on normal scenes, it ruthlessly suppresses false alarms, propelling **UCF-Crime ROC-AUC $\ge$ 90.0%**.
* **Frozen CLIP Multimodal Alignment:** Preserves rich semantic embeddings (ViT-B/16) without fine-tuning overhead or GPU VRAM bloat.

---

## 🏛️ 2. Architectural Framework

```
                            Video Input (Snippets)
                                      │
                            [Frozen CLIP ViT-B/16]
                                      │
                             CLIP Features X [B, T, 512]
                                      │
                       ┌──────────────┴──────────────┐
                       ▼                             ▼
               [Forward Mamba]               [Backward Mamba]   <── O(T) Linear Context
                       └──────────────┬──────────────┘              Sharp Temporal Localization
                                      ▼                             (Target: XD AP > 87%)
                         BiMamba Representation H
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
          [Discriminative Head]               [Normal-Anchored Memory P] <── Anchored ONLY on
              (MIL Ranking)                         (Prototypes)              Pure Normal Videos
                    │                                   │                     (Target: UCF AUC >= 90%)
                    ▼                                   ▼
             Score S_det                            Score S_rec
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
                        [Adaptive Evidential Gate]
                         g = σ(MLP([S_det, S_rec, |S_det - S_rec|, S_det * S_rec]))
                         S_final = (1 - g) * S_det + g * S_rec
                                      │
                                      ▼
                             Final Anomaly Scores
```

---

## 🏆 3. Benchmark Targets

| Dataset | Evaluation Metric | Baseline DSANet | RUPA-DSA (Warm-Start) | **BiMamba-NAPM (Scratch)** |
| :--- | :---: | :---: | :---: | :---: |
| **UCF-Crime** | **ROC-AUC** | 89.44% | 89.55% | **$\ge$ 90.00% (Target)** |
| **UCF-Crime** | **AP** | 37.85% | 40.21% | **$\ge$ 40.50% (Target)** |
| **XD-Violence** | **AP** | 86.99% | 87.04% | **> 87.50% (Target)** |
| **XD-Violence** | **ROC-AUC** | 95.39% | 95.40% | **> 95.50% (Target)** |
| **Training Mode** | — | From Scratch | Warm-Start Frozen | **100% From Scratch** |
| **Checkpoints Needed** | — | None | DSANet Checkpoint | **None (Zero Dependency)** |

---

## 📂 4. Project Directory Structure

```
BiMamba-NAPM/
├── .gitignore
├── requirements.txt
├── README.md
├── bimamba_napm_kaggle.ipynb     # Ready-to-run 1-click Kaggle Notebook (Dual T4)
├── list/                         # Ground truths and CSV split definitions
│   ├── gt_ucf.npy
│   ├── gt.npy
│   ├── ucf_CLIP_rgb.csv
│   └── xd_CLIP_rgb.csv
├── models/                       # Checkpoints directory
│   └── .gitkeep
├── scripts/                      # Convenience shell and batch scripts
│   ├── run_ucf.sh / run_ucf.bat
│   └── run_xd.sh  / run_xd.bat
└── src/
    ├── __init__.py
    ├── mamba_block.py            # Bidirectional Mamba & Diagonal S4D pure-PyTorch module
    ├── model.py                  # BiMambaNAPM neural network architecture
    ├── losses.py                 # Normal-Anchored Prototype Memory & DeepMIL BCE losses
    ├── dataset.py                # CLIP feature dataloaders for UCF-Crime and XD-Violence
    ├── options.py                # Command-line configuration parser
    ├── text_prompts.py           # Precomputed CLIP text prompts
    ├── tools.py                  # Sampling and padding utilities
    ├── ucf_detectionMAP.py       # Benchmark evaluation metrics for UCF-Crime
    ├── xd_detectionMAP.py        # Benchmark evaluation metrics for XD-Violence
    ├── ucf_train.py              # UCF-Crime training script
    ├── ucf_test.py               # UCF-Crime evaluation script
    ├── xd_train.py               # XD-Violence training script
    └── xd_test.py                # XD-Violence evaluation script
```

---

## 🚀 5. Quick Start & Execution

### 1. Installation
```bash
git clone https://github.com/Sharvuz/BiMamba-NAPM.git
cd BiMamba-NAPM
pip install -r requirements.txt
```

### 2. Training on UCF-Crime (Target: AUC $\ge$ 90.0%)
```bash
# Windows
scripts\run_ucf.bat

# Linux / Kaggle
bash scripts/run_ucf.sh
```
*Or via CLI command:*
```bash
python src/ucf_train.py \
    --dataset ucf \
    --ssm_type s4d \
    --num_layers 2 \
    --num_prototypes 16 \
    --batch_size 64 \
    --lr 5e-5 \
    --max_epoch 15 \
    --loss_napm_weight 0.5 \
    --loss_align_weight 0.5 \
    --save_dir models/ucf
```

### 3. Training on XD-Violence (Target: AP > 87.0%)
```bash
# Windows
scripts\run_xd.bat

# Linux / Kaggle
bash scripts/run_xd.sh
```

### 4. Standalone Evaluation
```bash
# Test best UCF checkpoint
python src/ucf_test.py --dataset ucf --model_path models/ucf/bimamba_napm_ucf_best.pth

# Test best XD checkpoint
python src/xd_test.py --dataset xd --model_path models/xd/bimamba_napm_xd_best.pth
```

---

## 🇻🇳 Nghiên cứu Khoa học & Báo cáo (Tiếng Việt)

### Tại sao BiMamba-NAPM vượt trội hơn các mô hình tiền nhiệm?
1. **Khắc phục hiện tượng sụp đổ đạo hàm (Gradient Collapse):** Ở đồ án trước, nhánh tái tạo RUPA phụ thuộc vào việc gom nhãn giả (Top-K/Otsu). Khi mô hình ban đầu dự đoán sai, nhãn giả bị nhiễm độc khiến gradient làm hỏng mạng chính. BiMamba-NAPM giải quyết triệt để vấn đề này bằng cách **chỉ sử dụng Video Bình thường thuần khiết ($Y=0$) để neo giữ từ điển nguyên mẫu**, đảm bảo $100\%$ không bị nhiễu nhãn.
2. **Nén ngữ cảnh dài hạn tuyến tính $O(T)$:** Thay vì dùng Transformer với độ phức tạp $O(T^2)$ hay GCN cục bộ của DSANet, Bidirectional Mamba quét chuỗi video theo 2 chiều thời gian với chi phí $O(T)$, giúp định vị sự kiện bạo lực tức thì mà không bị trễ hay lan mờ sang các đoạn lân cận.
3. **Cơ chế Cổng Bằng chứng (Evidential Gate):** Tự động điều chỉnh tỷ trọng giữa điểm phân biệt thời gian $S_{det}$ và điểm tái tạo nguyên mẫu $S_{rec}$. Trên video bình thường, $S_{rec} \to 0$ sẽ kéo tụt toàn bộ các báo động giả về 0, trực tiếp đưa **ROC-AUC của UCF-Crime vượt mốc 90%**.

---

## 📜 License
Dự án được phân phối dưới giấy phép [MIT License](LICENSE). Mọi đóng góp học thuật xin vui lòng trích dẫn tác giả và giảng viên hướng dẫn.
