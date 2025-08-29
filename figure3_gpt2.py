import os, json, glob
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = r"D:/yangsy/results/gpt2_5pct_top3_new"
SAVE_DIR = os.path.join(OUTPUT_DIR, "plots_custom")
os.makedirs(SAVE_DIR, exist_ok=True)

DETAIL_GLOB = os.path.join(OUTPUT_DIR, "gen_[0-9][0-9][0-9][0-9][0-9][0-9].json")

xs, ys = [], []   # x: source relative start, y: suspicious relative start

for fp in sorted(glob.glob(DETAIL_GLOB)):
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    story_len = int(data.get("story_chars", 0) or 1)
    if story_len <= 0:
        continue

    for cand in data.get("candidates", []):
        segs = [s for s in cand.get("segments", []) if isinstance(s, dict)]
        if not segs:
            continue

        src_len = max((s.get("source_offset", [0,0])[1] for s in segs), default=0)
        if src_len <= 0:
            continue

        for s in segs:
            so = s.get("source_offset", [0, 0])
            sp = s.get("suspicious_offset", [0, 0])
            if not (isinstance(so, list) and len(so) == 2 and so[1] > so[0]):
                continue
            if not (isinstance(sp, list) and len(sp) == 2 and sp[1] > sp[0]):
                continue

            x = np.clip(so[0] / src_len, 0.0, 1.0)
            y = np.clip(sp[0] / story_len, 0.0, 1.0)
            xs.append(x); ys.append(y)

xs = np.array(xs, dtype=float)
ys = np.array(ys, dtype=float)
print(f"[INFO] collected segments: {len(xs)}")

if len(xs) > 0:
    fig, ax = plt.subplots(figsize=(6.8, 6.8))

    bins = 50
    H, xedges, yedges = np.histogram2d(xs, ys, bins=bins, range=[[0,1],[0,1]])

    im = ax.imshow(
        H.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap="Blues",    
        interpolation="nearest"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Count", rotation=90)

    ax.set_xlabel("Source relative position")
    ax.set_ylabel("Suspicious relative position")
    ax.set_title("Offset heatmap (normalized to [0,1])")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(False)

    fig.tight_layout()
    out_path = os.path.join(SAVE_DIR, "offset_heatmap_blues.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[OK] Saved heatmap to: {out_path}")
else:
    print("[WARN] No segments collected; heatmap not generated.")