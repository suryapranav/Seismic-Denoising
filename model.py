"""U-Net for STFT-domain denoising (Section II-C, Fig. 1).

Follows the authors' released code where the paper is ambiguous:
- 4 encoder levels + bottleneck, filters 64-128-256-256 | 512
- one 3x3 conv (stride 1) + LeakyReLU(0.1) per level, 2x2 max-pooling
- decoder: 2x2 transposed conv (stride 2), concatenate skip, 3x3 conv, dropout on the two deepest levels
- linear output layer, input zero-padded to a multiple of 16 and output cropped back

`width` scales every layer: width=64 is the paper model; width=16 is the quarter-size
model planned for edge deployment (Phase 2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, width=64, dropout=0.3):
        super().__init__()
        c1, c2, c3, c4, cb = width, 2 * width, 4 * width, 4 * width, 8 * width

        self.enc1 = nn.Conv2d(in_ch, c1, 3, padding=1)
        self.enc2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.enc3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.enc4 = nn.Conv2d(c3, c4, 3, padding=1)
        self.bottleneck = nn.Conv2d(c4, cb, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.up1, self.dec1 = nn.ConvTranspose2d(cb, c4, 2, stride=2), nn.Conv2d(2 * c4, c4, 3, padding=1)
        self.up2, self.dec2 = nn.ConvTranspose2d(c4, c3, 2, stride=2), nn.Conv2d(2 * c3, c3, 3, padding=1)
        self.up3, self.dec3 = nn.ConvTranspose2d(c3, c2, 2, stride=2), nn.Conv2d(2 * c2, c2, 3, padding=1)
        self.up4, self.dec4 = nn.ConvTranspose2d(c2, c1, 2, stride=2), nn.Conv2d(2 * c1, c1, 3, padding=1)
        self.out = nn.Conv2d(c1, out_ch, 3, padding=1)

        self.drop = nn.Dropout2d(dropout)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = (-h) % 16, (-w) % 16                       # 78 -> 80 for four 2x poolings
        x = F.pad(x, (0, pw, 0, ph))

        e1 = self.act(self.enc1(x))
        e2 = self.act(self.enc2(self.pool(e1)))
        e3 = self.act(self.enc3(self.pool(e2)))
        e4 = self.act(self.enc4(self.pool(e3)))
        b = self.act(self.bottleneck(self.pool(e4)))

        d = self.act(self.up1(b))
        d = self.drop(self.act(self.dec1(torch.cat([d, e4], 1))))
        d = self.act(self.up2(d))
        d = self.drop(self.act(self.dec2(torch.cat([d, e3], 1))))
        d = self.act(self.up3(d))
        d = self.act(self.dec3(torch.cat([d, e2], 1)))
        d = self.act(self.up4(d))
        d = self.act(self.dec4(torch.cat([d, e1], 1)))
        return self.out(d)[..., :h, :w]                      # crop back to 78 x 78


def sparse_loss(pred, target, alpha):
    """Eq. (6): (1 - alpha) * MSE + alpha * L1 norm of the prediction.

    L1 is a mean over elements, as in the authors' code (the paper writes a sum),
    so alpha values are comparable with theirs.
    """
    return (1 - alpha) * F.mse_loss(pred, target) + alpha * pred.abs().mean()


def count_params(model):
    return sum(p.numel() for p in model.parameters())
