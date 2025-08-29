import os, json, glob
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

OUTPUT_DIR = r"D:/yangsy/results/gpt2_5pct_top3_new"
VALID_TYPES = {"verbatim", "paraphrase", "summary"}  
SAVE_DIR = os.path.join(OUTPUT_DIR, "plots_custom")
os.makedirs(SAVE_DIR, exist_ok=True)

def iou(span_a, span_b):
    a0, a1 = span_a
    b0, b1 = span_b
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0

def merge_spans(spans, iou_th=0.5):
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    merged = []
    for s in spans:
        if not merged:
            merged.append([s[0], s[1]])
        else:
            a, b = merged[-1]
            if iou((a, b), (s[0], s[1])) >= iou_th or s[0] <= b:
                merged[-1][1] = max(b, s[1])
            else:
                merged.append([s[0], s[1]])
    return [(a, b) for a, b in merged]

detail_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "gen_[0-9][0-9][0-9][0-9][0-9][0-9].json")))

coverages = []                                  
type_counts = Counter()                         
type_coverages_per_story = defaultdict(list)    
stories_total = 0

for fp in detail_files:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    stories_total += 1
    L = int(data.get("story_chars", 0) or 1)

    all_spans = []
    spans_by_type = defaultdict(list)

    for cand in data.get("candidates", []):
        for seg in cand.get("segments", []):
            t = seg.get("type", "none")
            if t not in VALID_TYPES:
                continue
            s = seg.get("suspicious_offset", [0, 0])
            if not (isinstance(s, list) and len(s) == 2 and s[1] > s[0]):
                continue
            all_spans.append(tuple(s))
            spans_by_type[t].append(tuple(s))
            type_counts[t] += 1

    merged_all = merge_spans(all_spans, iou_th=0.5) if all_spans else []
    story_cov = sum(b - a for a, b in merged_all) / L
    coverages.append(min(1.0, story_cov))

    for t, lst in spans_by_type.items():
        mm = merge_spans(lst, iou_th=0.5) if lst else []
        cov_t = (sum(b - a for a, b in mm) / L) if L else 0.0
        type_coverages_per_story[t].append(cov_t)

agg_path = os.path.join(OUTPUT_DIR, "aggregate.json")
agg = {}
if os.path.exists(agg_path):
    with open(agg_path, "r", encoding="utf-8") as f:
        agg = json.load(f)

def annotate_hist(ax, counts, bins, fmt="{:d}", fontsize=8, y_offset=0.5):
    for count, left, right in zip(counts, bins[:-1], bins[1:]):
        if count > 0:
            x = (left + right) / 2
            ax.text(x, count + y_offset, fmt.format(int(count)),
                    ha="center", va="bottom", fontsize=fontsize)

def annotate_bars(ax, bars, fmt="{:.3f}", fontsize=9, y_offset=0.0):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + y_offset, fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize)

fig, ax = plt.subplots()
counts, bins, patches = ax.hist(coverages, bins=20, edgecolor="black")
ax.set_xlabel("Coverage ratio per story")
ax.set_ylabel("Count of stories")
ax.set_title("Coverage distribution (GPT-2 outputs)")
annotate_hist(ax, counts, bins, fmt="{:d}", fontsize=8, y_offset=max(counts)*0.02 if counts.size else 0.5)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "coverage_hist.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots()
ax.boxplot(coverages, vert=True, showmeans=True)
ax.set_ylabel("Coverage ratio per story")
ax.set_title("Coverage boxplot (GPT-2 outputs)")
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "coverage_boxplot.png"), dpi=150)
plt.close(fig)

labels = ["verbatim", "paraphrase", "summary"]
sizes = [type_counts.get(k, 0) for k in labels]
if sum(sizes) == 0:  
    sizes = [1, 0, 0]
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct="%.1f%%", startangle=90)
ax.set_title("Segment type proportion (by count)")
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "type_pie.png"), dpi=150)
plt.close(fig)

mean_cov_by_type = []
for k in labels:
    vals = type_coverages_per_story.get(k, [])
    mean_cov_by_type.append(sum(vals) / len(vals) if vals else 0.0)

fig, ax = plt.subplots()
bars = ax.bar(labels, mean_cov_by_type)
ax.set_ylabel("Mean coverage by type")
ax.set_title("Coverage comparison by type (GPT-2 outputs)")
annotate_bars(ax, bars, fmt="{:.3f}", fontsize=9, y_offset=max(mean_cov_by_type)*0.02 if any(mean_cov_by_type) else 0.005)
fig.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, "type_coverage_bar.png"), dpi=150)
plt.close(fig)

print("[OK] Saved figures to:", SAVE_DIR)
stories_positive = sum(c > 0 for c in coverages)
print(f"Stories total: {stories_total}")
print(f"Stories positive: {stories_positive}  | Story-level rate: {stories_positive / max(1, stories_total):.3f}")
print(f"Mean coverage (all stories): {sum(coverages)/max(1,len(coverages)):.4f}")
print("Segment count by type:", dict(type_counts))
print("Mean coverage by type:", dict(zip(labels, mean_cov_by_type)))