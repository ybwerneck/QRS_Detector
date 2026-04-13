"""ECG mask model — HuBERT-ECG encoder + compressor/fusion head.

Input  : (N, 12, 2500)  — 12-lead, 500 Hz, 5-second tiled window
         (N, W)         — Pan-Tompkins decision signal (1:1 ms), W = WINDOW_PRE+WINDOW_POST
Output : (N, 2, W)      — soft probability mask [QRS, QT] over the beat window
                          sum over last dim ≈ duration in ms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Positive(nn.Module):
    """Parametrization that constrains a weight tensor to be strictly positive via softplus."""
    def forward(self, x):
        return F.softplus(x)


class MaskHead(nn.Module):
    """Compressor + fusion head: HuBERT embeddings + PT signal → (2, 550) mask.

    Architecture
    ------------
    g path  — COMPRESSOR
      (N, 12, t, D) permute→ (N, D, 12, t)
      Conv2d bottleneck + depthwise lead-collapse → (N, 1, 1, t)
      squeeze + upsample → (N, 1, window_size)   coarse prior g

    f path  — POINTWISE MLP on PT decision signal
      (N, window_size) → (N, 1, window_size)     refined prior f

    FUSION
      cat[sigmoid(f), sigmoid(g)] → (N, 2, window_size)
      Conv1d × 3 → (N, 2, window_size) logits   (2 channels: QRS, QT)
      sigmoid → mask
    """

    def __init__(self, embed_dim=768, window_size=None, width=128):
        # width is accepted for API compatibility but unused
        super().__init__()
        if window_size is None:
            from beat import WINDOW_PRE, WINDOW_POST
            window_size = WINDOW_PRE + WINDOW_POST
        self.window_size = window_size

        self.compress = nn.Sequential(
            nn.Conv2d(embed_dim, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.GELU(),                                                            # (N, 1024, 12, t)

            nn.Conv2d(1024, 512, kernel_size=1), nn.GELU(),                        # (N, 512, 12, t)
            
            nn.Conv2d(512, 128, kernel_size=(1, 7), padding=(0, 3), groups=128), nn.GELU(),  # (N, 128, 12, t)
            nn.Conv2d(128, 64,  kernel_size=(1, 7), padding=(0, 3), groups=64),  nn.GELU(),  # (N, 64,  12, t)
            nn.Conv2d(64,  32,  kernel_size=(1, 7), padding=(0, 3), groups=32),  nn.GELU(),  # (N, 32,  12, t)
            nn.Conv2d(32,  16,  kernel_size=(1, 7), padding=(0, 3), groups=16),  nn.GELU(),  # (N, 16,  12, t)

            # learned lead collapse at the end
            nn.Conv2d(16,   1,  kernel_size=(12, 7), padding=(0, 3)),                         # (N,  1,  1, t)
        )
        # ── f path: pointwise MLP on PT signal ───────────────────────
        self.pt_mlp = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=1), nn.GELU(),

            nn.Conv1d(128, 1, kernel_size=1),                # (N, 1, 550)
        )

        # ── fusion ────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3), nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=7, padding=3), nn.GELU(),
            nn.Conv1d(64, 1, kernel_size=1),                 # (N, 1, 550) logits — QRS only
        )

        # ablation flags
        self.ablate_embedding = False
        self.ablate_decision  = False

    def _forward_impl(self, x, decision):
        # x: (N, L, t, D) — e.g. (N, 12, 96, 768)
        N, L, t, D = x.shape

        # ── g: compressor ─────────────────────────────────────────────
        g = x.permute(0, 3, 1, 2)          # (N, D, L, t) = (N, 768, 12, 96)
        if self.ablate_embedding:
            g = torch.zeros_like(g)
        g = self.compress(g)                # (N, 1, 1, t)
        g = g.squeeze(2)                    # (N, 1, t)
        g = F.interpolate(g, size=self.window_size,
        mode='linear', align_corners=False)  # (N, 1, 550)





        # ── f: PT signal ──────────────────────────────────────────────
        d = decision.unsqueeze(1)           # (N, 1, 550)
        if self.ablate_decision:
            d = torch.zeros_like(d)
        
        f = d#self.pt_mlp(d)                  # (N, 1, 550)
        mu  = d.mean(dim=-1, keepdim=True)
        std = d.std(dim=-1, keepdim=True).clamp(min=1e-6)
        f = (d - mu) / std   
        
        mu  = g.mean(dim=-1, keepdim=True)
        std = g.std(dim=-1, keepdim=True).clamp(min=1e-6)
        g = (g ) / std   
        

        logits =self.fusion(torch.cat([g, f], dim=1))  # (N, 1, 550)
        mask   = torch.sigmoid(logits)

        return logits, mask, mask.sum(dim=-1), f, g

    def forward(self, x, decision):
        logits, mask, durations, _, _ = self._forward_impl(x, decision)
        return logits, mask, durations

    def forward_debug(self, x, decision):
        """Like forward but also returns f_sig and g_sig (pre-fusion branch outputs)."""
        return self._forward_impl(x, decision)

    def set_ablation(self, embedding=False, decision=False):
        self.ablate_embedding = embedding
        self.ablate_decision  = decision


class HuBERTECGRegressor(nn.Module):
    def __init__(
        self,
        repo_id='Edoardo-BS/hubert-ecg-base',
        width=128,
        freeze=True,
    ):
        super().__init__()
        from transformers import AutoModel
        from beat import WINDOW_PRE, WINDOW_POST

        self.encoder = AutoModel.from_pretrained(
            repo_id, trust_remote_code=True
        )

        embed_dim = self.encoder.config.hidden_size

        # infer t via dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 12, 2500)
            N, L, T = dummy.shape
            dummy_flat = dummy.reshape(N * L, T)
            out = self.encoder(input_values=dummy_flat).last_hidden_state
            self.t = out.shape[1]
            self.L = L  # 12

        self.head = MaskHead(
            embed_dim=embed_dim,
            window_size=WINDOW_PRE + WINDOW_POST,
            width=width,
        )

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def encode(self, x):
        N, L, T = x.shape
        x_flat = x.reshape(N * L, T)
        out = self.encoder(input_values=x_flat).last_hidden_state  # (N*L, t, D)
        t, D = out.shape[1], out.shape[2]
        return out.reshape(N, L, t, D)

    def forward(self, x, decision):
        feats = self.encode(x)              # (N, 12, t, D)
        return self.head(feats, decision)   # (mask, durations)


class PTHead(nn.Module):
    """Minimal Pan-Tompkins baseline head.

    No embeddings used — output is purely sigmoid(decision_signal).
    Matches the MaskHead interface exactly so it drops into the same
    training/eval loop without any changes.
    """

    def __init__(self, window_size=None, **kwargs):
        super().__init__()
        if window_size is None:
            from beat import WINDOW_PRE, WINDOW_POST
            window_size = WINDOW_PRE + WINDOW_POST
        self.window_size = window_size
                # ── f path: pointwise MLP on PT signal ───────────────────────
        self.scale=nn.Linear(window_size, window_size)
        nn.utils.parametrize.register_parametrization(self.scale, 'weight', _Positive())
        nn.utils.parametrize.register_parametrization(self.scale, 'bias',   _Positive())

    def _forward_impl(self, x, decision):
        # x (embeddings) intentionally ignored
        d      = decision.unsqueeze(1)          # (N, 1, W)
        mu  = d.mean(dim=-1, keepdim=True)
        std = d.std(dim=-1, keepdim=True).clamp(min=1e-6)
        f_sig = (d - mu) / std   
        f=self.scale(f_sig)
        mask   = torch.sigmoid(f)
        durations = mask.sum(dim=-1)                        # (N, 1)

        # for debug plot
        w = self.scale.weight
        g_sig  = (w.unsqueeze(0)) / (w.std(dim=-1, keepdim=True).clamp(min=1e-6))  # (1, W)


        return f, mask, durations, f_sig, g_sig

    def forward(self, x, decision):
        logits, mask, durations, _, _ = self._forward_impl(x, decision)
        return logits, mask, durations

    def forward_debug(self, x, decision):
        return self._forward_impl(x, decision)


def build_model(device=None, **kwargs):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HuBERTECGRegressor(**kwargs).to(device)
    return model, device


if __name__ == '__main__':
    from beat import process_study, WINDOW_PRE, WINDOW_POST
    from dataset import BeatDataset, preprocess_hubert, build_mask

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    beats, _, _ = process_study('data/p21/ecg_data.txt')
    ds = BeatDataset(beats, transform=preprocess_hubert)
    x, d, y_mask = ds[0]

    x      = x.unsqueeze(0).to(device)       # (1, 12, 2500)
    d      = d.unsqueeze(0).to(device)       # (1, 550)
    y_mask = y_mask.unsqueeze(0).to(device)  # (1, 2, 550)

    model, _ = build_model(device=device, freeze=True, width=256)
    n_params   = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params     : {n_params:,}')
    print(f'Trainable params : {n_trainable:,}')

    model.eval()
    with torch.no_grad():
        logits, mask, durations = model(x, d)   # (1,2,550), (1,2,550), (1,2)

    beat = ds.beats[0]
    true_dur = y_mask[0].sum(dim=-1)    # (2,)

    print(f'\nBeat  spike={beat.spike_idx} ms')
    print(f'QRS   pred={durations[0,0]:.1f} ms   true={true_dur[0]:.0f} ms   err={durations[0,0]-true_dur[0]:+.1f} ms')
    print(f'QT    pred={durations[0]:.1f} ms   true={true_dur[1]:.0f} ms   err={durations[0,1]-true_dur[1]:+.1f} ms')
    print(f'\nLogits shape    : {tuple(logits.shape)}')
    print(f'Mask shape      : {tuple(mask.shape)}')
    print(f'Durations shape : {tuple(durations.shape)}')
    print(f'Mask range      : [{mask.min():.3f}, {mask.max():.3f}]')
