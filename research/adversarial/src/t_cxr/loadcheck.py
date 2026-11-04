"""Compatibility check of the released PriCheXy-Net weights with our re-implementations (no ChestX-ray14 data used)."""
import sys
import torch
from . import attackers as A, generator as G

W = sys.argv[1]
for mu in ("0.001", "0.005", "0.01"):
    sd = torch.load(f"{W}/generator_lowest_total_loss_mu_{mu}.pth", map_location="cpu")
    g = G.UNet(1, 2, 32)
    r = g.load_state_dict(sd, strict=True)
    print("generator mu", mu, "strict load ok;", sum(p.numel() for p in g.parameters()) / 1e6, "M params; BN buffers present:", any("running_mean" in k for k in sd))
sd = torch.load(f"{W}/pretrained_generator_prichexy_net.pth", map_location="cpu"); g = G.UNet(1, 2, 32); g.load_state_dict(sd); g.eval()
x = torch.rand(4, 1, 256, 256)
with torch.no_grad():
    f = g(x)
print("pretrained (identity-reconstruction) generator on noise: |raw flow| mean %.4f max %.4f  (noise input: indicative only; real-image statistics are measured after the data gate)" % (f.abs().mean(), f.abs().max()))
sd = torch.load(f"{W}/pretrained_verification_model.pth", map_location="cpu")
s = A.Siamese("resnet50", pretrained=False)
print("verification model strict load:", s.load_state_dict(sd, strict=True))
m = torch.load(f"{W}/pretrained_classifier.pth", map_location="cpu", weights_only=False)
print("classifier file keys:", list(m.keys()) if isinstance(m, dict) else type(m), "| model type", type(m["model"]).__name__, "| last layer:", m["model"].classifier)
