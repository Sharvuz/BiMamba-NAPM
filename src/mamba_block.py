import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as OfficialMamba
    HAS_OFFICIAL_MAMBA = True
except ImportError:
    HAS_OFFICIAL_MAMBA = False


class PurePyTorchMamba(nn.Module):
    """
    Pure-PyTorch implementation of the Selective State Space Model (Mamba).
    Requires zero CUDA C++ compilation, fully portable across Windows/Linux/CPU/GPU.
    """
    def __init__(self, d_model=512, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # A matrix log initialized (HiPPO-like)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        
        # 1D Temporal Convolution
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_act = F.silu(x_conv)
        
        # SSM Projections
        ssm_proj = self.x_proj(x_act)
        delta_raw, B_param, C_param = torch.split(
            ssm_proj, [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_raw))
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Discretization
        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, T, d_inner, d_state)
        dB = delta.unsqueeze(-1) * B_param.unsqueeze(2)                     # (B, T, d_inner, d_state)
        
        # Sequential State Space Scan
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        y_list = []
        for t in range(T):
            h = dA[:, t] * h + dB[:, t] * x_act[:, t, :, None]
            y_t = torch.sum(h * C_param[:, t, None, :], dim=-1)
            y_list.append(y_t)
            
        y = torch.stack(y_list, dim=1)  # (B, T, d_inner)
        y = y + x_act * self.D
        y = y * F.silu(z)
        out = self.dropout(self.out_proj(y))
        return out


class DiagonalSSM(nn.Module):
    """
    Diagonal State Space Model (S4D style).
    Fast non-recurrent parallel computation using 1D depthwise convolutions.
    Extremely fast for long video sequences during end-to-end training.
    """
    def __init__(self, d_model=512, d_state=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Diagonal A initialization
        self.log_step = nn.Parameter(torch.log(torch.tensor(0.001)))
        self.A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))
        self.C = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.D = nn.Parameter(torch.ones(d_model))
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, H = x.shape
        dt = torch.exp(self.log_step)
        A = -torch.exp(self.A_real)
        
        # Kernel computation over time
        t = torch.arange(T, device=x.device, dtype=torch.float32)
        k = torch.einsum('hn,t->hnt', A * dt, t)
        K = torch.einsum('hn,hnt->ht', self.C, torch.exp(k)) * dt  # (H, T)
        
        # Parallel convolution via FFT
        x_t = x.transpose(1, 2)  # (B, H, T)
        K_f = torch.fft.rfft(K, n=2*T)
        x_f = torch.fft.rfft(x_t, n=2*T)
        y = torch.fft.irfft(x_f * K_f, n=2*T)[..., :T]
        
        y = y + x_t * self.D.unsqueeze(-1)
        out = self.dropout(self.out_proj(y.transpose(1, 2)))
        return out


class BiMambaBlock(nn.Module):
    """
    Bidirectional State Space Block (Forward + Backward Scanning).
    Captures full past and future temporal context with O(T) complexity.
    """
    def __init__(self, d_model=512, d_state=16, ssm_type='s4d', expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.ssm_type = ssm_type
        self.norm = nn.LayerNorm(d_model)

        if ssm_type == 'official' and HAS_OFFICIAL_MAMBA:
            self.fwd_ssm = OfficialMamba(d_model=d_model, d_state=d_state, d_conv=4, expand=expand)
            self.bwd_ssm = OfficialMamba(d_model=d_model, d_state=d_state, d_conv=4, expand=expand)
        elif ssm_type == 's4d':
            self.fwd_ssm = DiagonalSSM(d_model=d_model, d_state=d_state * 2, dropout=dropout)
            self.bwd_ssm = DiagonalSSM(d_model=d_model, d_state=d_state * 2, dropout=dropout)
        else:
            self.fwd_ssm = PurePyTorchMamba(d_model=d_model, d_state=d_state, expand=expand, dropout=dropout)
            self.bwd_ssm = PurePyTorchMamba(d_model=d_model, d_state=d_state, expand=expand, dropout=dropout)

        self.merge = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        residual = x
        x_norm = self.norm(x)

        if padding_mask is not None:
            x_norm = x_norm.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Forward scan
        fwd_out = self.fwd_ssm(x_norm)

        # Backward scan (reverse along temporal dimension T)
        x_rev = torch.flip(x_norm, dims=[1])
        bwd_out = self.bwd_ssm(x_rev)
        bwd_out = torch.flip(bwd_out, dims=[1])

        # Merge bidirectional context
        merged = self.merge(torch.cat([fwd_out, bwd_out], dim=-1))
        x = residual + merged

        # Position-wise feed-forward
        x = x + self.ffn(self.norm2(x))
        return x
