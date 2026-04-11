"""ECG regression model — HuBERT-ECG encoder + MLP regression head.

Input  : (N, 12, 2500)  — 12-lead, 500 Hz, 5-second tiled window
Output : (N, 2)         — [qrs_duration_ms, qt_interval_ms]
"""

import torch
import torch.nn as nn


import torch
import torch.nn as nn
class LearnedCompressor(nn.Module):
    def __init__(self, embed_dim, k, L, t, width=256):
        super().__init__()

        time_stride = t // k
        W = width

        self.net = nn.Sequential(
            # 1. reduce channel dimension (D → W)
            nn.Conv2d(embed_dim, W, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(W),

            # 2. channel mixing at full width (4 layers ~263K at W=256)
            nn.Conv2d(W, W, kernel_size=1), nn.GELU(),
            nn.Conv2d(W, W, kernel_size=1), nn.GELU(),
            nn.Conv2d(W, W, kernel_size=1), nn.GELU(),
            nn.Conv2d(W, W, kernel_size=1), nn.GELU(),

            # 3. collapse leads (depthwise, cheap)
            nn.Conv2d(W, W, kernel_size=(L, 1), groups=W),
            nn.GELU(),

            # 4. collapse time (depthwise, cheap)
            nn.Conv2d(W, W, kernel_size=(1, time_stride),
                      stride=(1, time_stride), groups=W),
            nn.GELU(),

            # 5. output projection W → 64
            nn.Conv2d(W, 64, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)

class ConvHead(nn.Module):
    def __init__(self, embed_dim, embed_L, embed_t,
                 decision_size=530, hidden=256, dropout=0.1, k=1):
        super().__init__()

        self.k = k
        self.L=embed_L
        self.t=embed_t

        # -------------------------
        # Learned compression
        # -------------------------
        self.conv = LearnedCompressor(embed_dim, k, embed_L, embed_t)

        # -------------------------
        # Decision encoder  (530→512→256→256→64, ~485K)
        # -------------------------
        self.decision_enc = nn.Sequential(
            nn.Linear(decision_size, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
        )
        feat_dim = 64 * k + 64

        # -------------------------
        # QRS head
        # -------------------------
        self.qrs_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # -------------------------
        # QT head (QRS conditioned)
        # -------------------------
        self.qt_head = nn.Sequential(
            nn.Linear(feat_dim + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # controls
        self.qt_enabled = True
        self.detach_qrs_for_qt = False

        # ablation flags
        self.ablate_embedding = False   # zero out the HuBERT embedding branch
        self.ablate_decision  = False   # zero out the decision signal branch

    # -------------------------
    # Controls
    # -------------------------
    def enable_qt(self, flag=True):
        self.qt_enabled = flag

    def freeze_qrs(self):
        for p in self.conv.parameters():
            p.requires_grad = False
        for p in self.decision_enc.parameters():
            p.requires_grad = False
        for p in self.qrs_head.parameters():
            p.requires_grad = False

    def freeze_qt(self):
        for p in self.qt_head.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def set_detach_qrs(self, flag=True):
        self.detach_qrs_for_qt = flag

    def set_ablation(self, embedding=False, decision=False):
        self.ablate_embedding = embedding
        self.ablate_decision  = decision

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x, decision):
        # x: (N, L, t, D) → (N, D, L, t)
        x = x.permute(0, 3, 1, 2)

        # ---- Compression ----
        if self.ablate_embedding:
            x = torch.zeros_like(x)
        x = self.conv(x).flatten(1)                # (N, 64*k)

        # ---- Decision branch ----
        decision_feat = self.decision_enc(decision) # (N, 64)
        if self.ablate_decision:
            decision_feat = torch.zeros_like(decision_feat)

        feat = torch.cat([x, decision_feat], dim=1)

        # ---- QRS ----
        qrs = self.qrs_head(feat)

        # ---- QT ----
        if self.qt_enabled:
            qrs_for_qt = qrs.detach() if self.detach_qrs_for_qt else qrs
            qt_input = torch.cat([feat, qrs_for_qt], dim=1)
            qt = self.qt_head(qt_input)
        else:
            qt = torch.zeros_like(qrs)

        return torch.cat([qrs, qt], dim=1)
        
class HuBERTECGRegressor(nn.Module):
    def __init__(
        self,
        repo_id='Edoardo-BS/hubert-ecg-base',
        hidden=256,
        dropout=0.1,
        freeze=True,
    ):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(
            repo_id, trust_remote_code=True
        )

        embed_dim = self.encoder.config.hidden_size

        # ---- infer t via dummy forward ----
        with torch.no_grad():
            dummy = torch.zeros(1, 12, 2500)
            N, L, T = dummy.shape

            dummy_flat = dummy.reshape(N * L, T)
            out = self.encoder(input_values=dummy_flat).last_hidden_state

            self.t = out.shape[1]   # store it
            self.L = L              # = 12

        from beat import WINDOW_PRE, WINDOW_POST

        self.head = ConvHead(
            embed_dim,
            L,
            self.t,
            decision_size=WINDOW_PRE + WINDOW_POST,
            hidden=hidden,
            dropout=dropout,
        )

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def encode(self, x):
        N, L, T = x.shape

        x_flat = x.reshape(N * L, T)
        out = self.encoder(input_values=x_flat).last_hidden_state  # (N*L, t, D)

        t, D = out.shape[1], out.shape[2]
       # print(out.shape, flush=True)
        return out.reshape(N, L, t, D)

    def forward(self, x, decision):
        feats = self.encode(x)              # (N, 12, t, D)
        return self.head(feats, decision)

def build_model(device=None, **kwargs):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HuBERTECGRegressor(**kwargs).to(device)
    return model, device


if __name__ == '__main__':
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from beat import process_study
    from dataset import BeatDataset, preprocess_hubert

    LEAD_NAMES  = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    DEMO_LEAD   = 1   # Lead II — shown in the channel time-series panel
    N_CHANNELS  = 8   # number of embedding dims to trace

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    beats, _, _ = process_study('data/p21/ecg_data.txt')
    ds      = BeatDataset(beats, transform=preprocess_hubert)
    x, d, y = ds[0]
    x = x.unsqueeze(0).to(device)   # (1, 12, 2500)
    d = d.unsqueeze(0).to(device)   # (1, 530)

    model, _ = build_model(device=device)
    model.eval()
    with torch.no_grad():
        enc  = model.encode(x).squeeze(0).cpu().numpy()   # (12, t, D)
        pred = model(x, d).cpu()

    L, t, D = enc.shape
    print(f'Encoding : {enc.shape}  (leads × frames × dim)')
    print(f'Pred     : qrs={pred[0,0]:.1f} ms   qt={pred[0,1]:.1f} ms')
    print(f'Target   : qrs={y[0]:.1f} ms       qt={y[1]:.1f} ms')

    # ---- derived quantities -----------------------------------------
    per_lead_avg = enc.mean(axis=1)                          # (12, D)  time-averaged

    # lead-to-lead cosine similarity
    norms    = np.linalg.norm(per_lead_avg, axis=1, keepdims=True)
    normed   = per_lead_avg / (norms + 1e-8)
    cos_mat  = normed @ normed.T                             # (12, 12)

    # evenly-spaced embedding channels for the time-series panel
    ch_idx   = np.linspace(0, D - 1, N_CHANNELS, dtype=int)
    lead_seq = enc[DEMO_LEAD]                                # (t, D)
    ms_axis  = np.linspace(0, 5000, t)                      # approx ms (5 s window)

    # ---- Lead II — swap: 768 as "time", 38 frames as "channels" ----
    lead_II  = enc[DEMO_LEAD].T                           # (D, t) = (768, 38)
    dim_axis = np.arange(D)                               # x-axis: 0..767

    # dim-to-dim cosine similarity: each dim has a 38-frame vector
    norms_f    = np.linalg.norm(lead_II, axis=1, keepdims=True)  # (D, 1)
    normed_f   = lead_II / (norms_f + 1e-8)                      # (D, t)
    cos_frames = normed_f @ normed_f.T                            # kept for compat (D, D) — also used below

    # L2 norm per dim-position (over the 38 channels)
    dim_norms = np.linalg.norm(lead_II, axis=1)           # (D,)

    # a few evenly-spaced channels (original frames) to trace over 768 dims
    N_CH     = 8
    ch_pick  = np.linspace(0, t - 1, N_CH, dtype=int)

    # ---- layout -----------------------------------------------------
    fig = plt.figure(figsize=(17, 11))
    gs  = gridspec.GridSpec(
        3, 2,
        height_ratios=[1.4, 1.0, 1.2],
        hspace=0.55, wspace=0.35,
    )

    # 1. Heatmap  (38 frames × 768 dims) — channels on Y, dims on X
    ax0  = fig.add_subplot(gs[0, 0])
    vmax = np.percentile(np.abs(lead_II), 98)
    im0  = ax0.imshow(lead_II.T, aspect='auto', cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax0.set_xlabel('Embedding dimension', fontsize=8)
    ax0.set_ylabel('Frame (channel)', fontsize=8)
    ax0.set_title(f'{LEAD_NAMES[DEMO_LEAD]} embedding  ({t} frames × {D} dims)', fontsize=9)
    fig.colorbar(im0, ax=ax0, shrink=0.85, pad=0.02)

    # 2. Dim-to-dim cosine similarity  (768 × 768)
    cos_dims = normed_f @ normed_f.T                      # (D, D)
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(cos_dims, aspect='equal', cmap='coolwarm',
                     vmin=-1, vmax=1, interpolation='nearest')
    ax1.set_xlabel('Embedding dimension', fontsize=8)
    ax1.set_ylabel('Embedding dimension', fontsize=8)
    ax1.set_title('Dim-to-dim cosine similarity\n(using 38-frame vectors)', fontsize=9)
    fig.colorbar(im1, ax=ax1, shrink=0.85, pad=0.02)

    # 3. L2 norm over 768 dim-positions
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(dim_axis, dim_norms, color='steelblue', linewidth=1.2)
    ax2.fill_between(dim_axis, dim_norms, alpha=0.2, color='steelblue')
    ax2.set_xlabel('Embedding dimension', fontsize=8)
    ax2.set_ylabel('||emb||', fontsize=8)
    ax2.set_title(f'{LEAD_NAMES[DEMO_LEAD]} — L2 norm across 38 frames, per dim', fontsize=9)
    ax2.grid(alpha=0.25)

    # 4. Selected frames (channels) traced over 768 dims
    ax3    = fig.add_subplot(gs[2, :])
    colors = plt.cm.tab10(np.linspace(0, 1, N_CH))
    for fi, color in zip(ch_pick, colors):
        ax3.plot(dim_axis, lead_II[:, fi], color=color,
                 linewidth=1.2, label=f'frame {fi} (~{ms_axis[fi]:.0f}ms)')
    ax3.set_xlabel('Embedding dimension', fontsize=8)
    ax3.set_ylabel('Activation', fontsize=8)
    ax3.set_title(
        f'{LEAD_NAMES[DEMO_LEAD]} — {N_CH} evenly-spaced frames over 768 dims',
        fontsize=9,
    )
    ax3.legend(ncol=N_CH, fontsize=8, loc='upper right', framealpha=0.7)
    ax3.axhline(0, color='k', linewidth=0.6, linestyle='--', alpha=0.4)
    ax3.grid(alpha=0.25)

    fig.suptitle(
        f'HuBERT-ECG encoding  |  spike={beats[0].spike_idx} ms  '
        f'QRS={y[0]:.0f} ms  QT={y[1]:.0f} ms',
        fontsize=11,
    )
    plt.savefig('embedding_viz.png', dpi=150, bbox_inches='tight')
    print('Saved embedding_viz.png')
    model, _ = build_model(device=device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Total parameters: {n_params:,}')
    print(f'Trainable parameters: {n_trainable:,}')