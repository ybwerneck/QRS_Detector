"""ECG regression model — HuBERT-ECG encoder + MLP regression head.

Input  : (N, 12, 2500)  — 12-lead, 500 Hz, 5-second tiled window
Output : (N, 2)         — [qrs_duration_ms, qt_interval_ms]
"""

import torch
import torch.nn as nn


import torch
import torch.nn as nn


class ConvHead(nn.Module):
    def __init__(self, embed_dim, decision_size=530, hidden=256, dropout=0.1, k=8):
        super().__init__()

        self.k = k

        # -------------------------
        # Decision → FiLM (cheap)
        # -------------------------
        self.film = nn.Sequential(
            nn.LayerNorm(decision_size),
            nn.Linear(decision_size, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim * 2),  # gamma + beta
        )

        # -------------------------
        # Conv branch (temporal preserved)
        # -------------------------
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),

            nn.AdaptiveAvgPool2d((1, k)),  # ← key change
        )

        # -------------------------
        # Decision encoder (improved)
        # -------------------------
        self.decision_enc = nn.Sequential(
            nn.LayerNorm(decision_size),
            nn.Unflatten(1, (1, decision_size)),

            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.GELU(),

            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),  # (N, 64)
        )

        feat_dim = 64 * k + 64

        # -------------------------
        # QRS head
        # -------------------------
        self.qrs_head = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # -------------------------
        # QT head (QRS conditioned)
        # -------------------------
        self.qt_head = nn.Sequential(
            nn.Linear(feat_dim + 1, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # controls
        self.qt_enabled = False
        self.detach_qrs_for_qt = True

    # -------------------------
    # Controls (unchanged)
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
        for p in self.film.parameters():
            p.requires_grad = False

    def freeze_qt(self):
        for p in self.qt_head.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def set_detach_qrs(self, flag=True):
        self.detach_qrs_for_qt = flag

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x, decision):
        # x: (N, L, t, D) → (N, D, L, t)
        x = x.permute(0, 3, 1, 2)

        # ---- FiLM conditioning ----
        film_params = self.film(decision)              # (N, 2*D)
        gamma, beta = film_params.chunk(2, dim=1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)      # (N, D, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)

        x = gamma * x + beta

        # ---- Conv branch ----
        x = self.conv(x).flatten(1)                    # (N, 64*k)

        # ---- Decision branch ----
        d = self.decision_enc(decision)                # (N, 64)

        feat = torch.cat([x, d], dim=1)

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

        from beat import WINDOW_PRE, WINDOW_POST
        self.head = ConvHead(embed_dim,
                             decision_size=WINDOW_PRE + WINDOW_POST,
                             hidden=hidden, dropout=dropout)

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

    # ---- layout -----------------------------------------------------
    fig = plt.figure(figsize=(17, 11))
    gs  = gridspec.GridSpec(
        3, 2,
        height_ratios=[1.4, 1.0, 1.2],
        hspace=0.55, wspace=0.35,
    )

    # 1. Per-lead time-averaged embedding heatmap  (12 × D)
    ax0  = fig.add_subplot(gs[0, 0])
    vmax = np.percentile(np.abs(per_lead_avg), 98)
    im0  = ax0.imshow(per_lead_avg, aspect='auto', cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax0.set_yticks(range(L))
    ax0.set_yticklabels(LEAD_NAMES, fontsize=8)
    ax0.set_xlabel('Embedding dimension', fontsize=8)
    ax0.set_title(f'Per-lead mean embedding  (12 × {D})', fontsize=9)
    fig.colorbar(im0, ax=ax0, shrink=0.85, pad=0.02)

    # 2. Lead-to-lead cosine similarity matrix
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(cos_mat, aspect='equal', cmap='coolwarm',
                     vmin=-1, vmax=1, interpolation='nearest')
    ax1.set_xticks(range(L))
    ax1.set_xticklabels(LEAD_NAMES, rotation=45, ha='right', fontsize=7)
    ax1.set_yticks(range(L))
    ax1.set_yticklabels(LEAD_NAMES, fontsize=7)
    for i in range(L):
        for j in range(L):
            ax1.text(j, i, f'{cos_mat[i, j]:.2f}',
                     ha='center', va='center', fontsize=5,
                     color='white' if abs(cos_mat[i, j]) > 0.6 else 'black')
    ax1.set_title('Lead-to-lead cosine similarity', fontsize=9)
    fig.colorbar(im1, ax=ax1, shrink=0.85, pad=0.02)

    # 3. Per-frame L2 norm heatmap  (12 leads × t frames)
    frame_norms = np.linalg.norm(enc, axis=2)               # (12, t)
    ax2 = fig.add_subplot(gs[1, :])
    im2 = ax2.imshow(frame_norms, aspect='auto', cmap='viridis',
                     interpolation='nearest', extent=[0, 5000, L - 0.5, -0.5])
    ax2.set_yticks(range(L))
    ax2.set_yticklabels(LEAD_NAMES, fontsize=8)
    ax2.set_xlabel('Approx. time (ms)', fontsize=8)
    ax2.set_title('Embedding L2 norm over time — all leads', fontsize=9)
    fig.colorbar(im2, ax=ax2, shrink=0.6, pad=0.01, label='||emb||')

    # 4. Selected embedding channels over time for one lead
    ax3    = fig.add_subplot(gs[2, :])
    colors = plt.cm.tab10(np.linspace(0, 1, N_CHANNELS))
    for i, ci in enumerate(ch_idx):
        ax3.plot(ms_axis, lead_seq[:, ci],
                 color=colors[i], linewidth=1.5, label=f'dim {ci}')
    ax3.set_xlabel('Approx. time (ms)', fontsize=8)
    ax3.set_ylabel('Activation', fontsize=8)
    ax3.set_title(
        f'Embedding channels over time — {LEAD_NAMES[DEMO_LEAD]}'
        f'  ({N_CHANNELS} evenly-spaced dims out of {D})',
        fontsize=9,
    )
    ax3.legend(ncol=N_CHANNELS, fontsize=8, loc='upper right',
               framealpha=0.7)
    ax3.axhline(0, color='k', linewidth=0.6, linestyle='--', alpha=0.4)
    ax3.grid(alpha=0.25)

    plt.suptitle(
        f'HuBERT-ECG encoding  |  spike={beats[0].spike_idx} ms  '
        f'QRS={y[0]:.0f} ms  QT={y[1]:.0f} ms',
        fontsize=11, y=1.01,
    )
    plt.savefig('embedding_viz.png', dpi=150, bbox_inches='tight')
    print('Saved embedding_viz.png')
