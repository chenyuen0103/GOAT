#!/usr/bin/env bash
set -euo pipefail

# Qualitative pipeline for Rotated-MNIST @ 60 degrees:
# 1) train a standalone VAE on rotated MNIST
# 2) run GOAT framework experiment (includes GOAT, GOATCW, Ours-FR, Ours-Nat/ETA)
# 3) summarize pairwise comparisons from curves JSONL

REPO_DIR="${REPO_DIR:-$HOME/GOAT}"
CONDA_ENV="${CONDA_ENV:-gda}"
GPU="${GPU:-0}"

ANGLE="${ANGLE:-60}"
SEEDS="${SEEDS:-0 1 2}"
GT_DOMAINS="${GT_DOMAINS:-1}"
GENERATED_DOMAINS="${GENERATED_DOMAINS:-3}"
NUM_WORKERS="${NUM_WORKERS:-2}"

LABEL_SOURCE="${LABEL_SOURCE:-pseudo}"
EM_MATCH="${EM_MATCH:-pseudo}"
EM_SELECT="${EM_SELECT:-bic}"

VAE_EPOCHS="${VAE_EPOCHS:-20}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-256}"
VAE_LATENT_DIM="${VAE_LATENT_DIM:-20}"
VAE_LR="${VAE_LR:-1e-3}"

cd "$REPO_DIR"

RUN_TAG="rot${ANGLE}_gt${GT_DOMAINS}_gen${GENERATED_DOMAINS}_${LABEL_SOURCE}_${EM_MATCH}_${EM_SELECT}"
OUT_DIR="analysis_outputs/qualitative_${RUN_TAG}"
mkdir -p "$OUT_DIR"

echo "[1/3] Training VAE on Rotated-MNIST (${ANGLE} deg)"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$GPU" \
conda run -n "$CONDA_ENV" python - <<PY
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

angle = int("$ANGLE")
out_dir = "$OUT_DIR"
epochs = int("$VAE_EPOCHS")
batch_size = int("$VAE_BATCH_SIZE")
z_dim = int("$VAE_LATENT_DIM")
lr = float("$VAE_LR")
seed = 0

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(nn.Module):
    def __init__(self, x_dim=28*28, h_dim=512, z_dim=20):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(x_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
        )
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, x_dim), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar

def loss_fn(xhat, x, mu, logvar):
    bce = F.binary_cross_entropy(xhat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation((angle, angle)),
])

train_ds = datasets.MNIST(root="./data/mnist", train=True, transform=transform, download=True)
test_ds  = datasets.MNIST(root="./data/mnist", train=False, transform=transform, download=True)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

model = VAE(z_dim=z_dim).to(device)
opt = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
for ep in range(1, epochs + 1):
    total = 0.0
    for x, _ in train_loader:
        x = x.to(device).view(x.size(0), -1)
        xhat, mu, logvar = model(x)
        loss = loss_fn(xhat, x, mu, logvar)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"[VAE] epoch {ep}/{epochs} loss={total/len(train_ds):.4f}")

# Save model
os.makedirs(out_dir, exist_ok=True)
ckpt = os.path.join(out_dir, f"vae_rot{angle}_z{z_dim}.pt")
torch.save(model.state_dict(), ckpt)

# Reconstruction grid for qualitative check
model.eval()
with torch.no_grad():
    x, _ = next(iter(test_loader))
    x = x[:16].to(device)
    xflat = x.view(x.size(0), -1)
    xhat, _, _ = model(xflat)
    xhat = xhat.view(-1, 1, 28, 28)
    grid_in  = utils.make_grid(x.cpu(), nrow=8, pad_value=1.0)
    grid_out = utils.make_grid(xhat.cpu(), nrow=8, pad_value=1.0)
    utils.save_image(grid_in,  os.path.join(out_dir, f"vae_rot{angle}_inputs.png"))
    utils.save_image(grid_out, os.path.join(out_dir, f"vae_rot{angle}_recons.png"))
print(f"[VAE] saved to {ckpt}")
PY

echo "[2/3] Running GOAT methods on Rotated-MNIST (${ANGLE} deg)"
for seed in $SEEDS; do
  echo "  -> seed=${seed}"
  log_base="qual_${RUN_TAG}_s${seed}"

  rm -f \
    "logs/mnist/s${seed}/target${ANGLE}/${log_base}.txt" \
    "logs/mnist/s${seed}/target${ANGLE}/${log_base}_curves.jsonl"

  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$GPU" \
  conda run -n "$CONDA_ENV" python experiment_refrac.py \
    --dataset mnist \
    --mnist-mode normal \
    --rotation-angle "$ANGLE" \
    --seed "$seed" \
    --gt-domains "$GT_DOMAINS" \
    --generated-domains "$GENERATED_DOMAINS" \
    --label-source "$LABEL_SOURCE" \
    --em-match "$EM_MATCH" \
    --em-select "$EM_SELECT" \
    --num-workers "$NUM_WORKERS" \
    --log-file "$log_base"
done

echo "[3/3] Building qualitative comparison summary"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$GPU" \
conda run -n "$CONDA_ENV" python - <<PY
import glob
import json
import os
import statistics as stats

angle = int("$ANGLE")
out_dir = "$OUT_DIR"
run_tag = "$RUN_TAG"

paths = sorted(glob.glob(f"logs/mnist/s*/target{angle}/qual_{run_tag}_s*_curves.jsonl"))
if not paths:
    raise SystemExit("No curves files found. Check previous step logs.")

def get_last_test(method_block):
    vals = method_block.get("test_curve", [])
    return float(vals[-1]) if vals else float("nan")

rows = []
for p in paths:
    with open(p, "r") as f:
        rec = json.loads(f.readlines()[-1])
    m = rec.get("methods", {})
    row = {
        "seed": int(rec.get("seed", -1)),
        "goat": get_last_test(m.get("goat", {})),
        "goatcw": get_last_test(m.get("goat_classwise", {})),
        "ours_fr": get_last_test(m.get("ours_fr", {})),
        "ours_nat": get_last_test(m.get("ours_eta", {})),
    }
    rows.append(row)

rows = sorted(rows, key=lambda r: r["seed"])

# Pairwise margins requested by user
# 1) GOATCW - GOAT
# 2) OURS-FR - GOATCW
# 3) OURS-Nat - GOATCW
for r in rows:
    r["goatcw_minus_goat"] = r["goatcw"] - r["goat"]
    r["oursfr_minus_goatcw"] = r["ours_fr"] - r["goatcw"]
    r["oursnat_minus_goatcw"] = r["ours_nat"] - r["goatcw"]

summary = {
    "n_seeds": len(rows),
    "mean_goatcw_minus_goat": stats.fmean(r["goatcw_minus_goat"] for r in rows),
    "mean_oursfr_minus_goatcw": stats.fmean(r["oursfr_minus_goatcw"] for r in rows),
    "mean_oursnat_minus_goatcw": stats.fmean(r["oursnat_minus_goatcw"] for r in rows),
}

os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "method_pairwise_summary.csv")
json_path = os.path.join(out_dir, "method_pairwise_summary.json")

with open(csv_path, "w") as f:
    f.write("seed,GOAT,GOATCW,OURS_FR,OURS_Nat,GOATCW-GOAT,OURS_FR-GOATCW,OURS_Nat-GOATCW\n")
    for r in rows:
        f.write(
            f"{r['seed']},{r['goat']:.6f},{r['goatcw']:.6f},{r['ours_fr']:.6f},{r['ours_nat']:.6f},"
            f"{r['goatcw_minus_goat']:.6f},{r['oursfr_minus_goatcw']:.6f},{r['oursnat_minus_goatcw']:.6f}\n"
        )

with open(json_path, "w") as f:
    json.dump({"per_seed": rows, "aggregate": summary}, f, indent=2)

print(f"Saved: {csv_path}")
print(f"Saved: {json_path}")
print("Aggregate:", summary)
PY

echo "Done. Outputs are under: $OUT_DIR"
echo "- VAE checkpoint/reconstructions"
echo "- Pairwise method summary (CSV/JSON)"
