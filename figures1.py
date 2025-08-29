# figures_pan2013.py
# 聚合 PAN2013 基准模式的真值与预测，生成 3 张图 + 导出对齐数据
import os
import re
import json
import pathlib
import logging
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import LinearSegmentedColormap

PAN_ROOT     = "C:/Users/YangSY/Desktop/homework/MSc_desk/pan13-text-alignment-test-corpus2-2013-01-21"
RESULTS_DIR  = "C:/Users/YangSY/Desktop/homework/MSc_desk/ALL_Results/pan2013_improve_outputs"
OUT_DIR      = "C:/Users/YangSY/Desktop/homework/MSc_desk/ALL_Results/pan2013_figures" 

os.makedirs(OUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

CLASSES = ["none", "verbatim", "paraphrase", "summary"]

COL_PRECISION = "#506D9C"  
COL_RECALL    = "#55A868"  
COL_F1        = "#C44E52"  
COL_MACRO     = "#4C72B0"  
COL_WEIGHTED  = "#8172B2"  


def get_ground_truth_mapping(pan_root: str):
    type_map = {
        "01-no-plagiarism": "none",
        "02-no-obfuscation": "verbatim",
        "03-random-obfuscation": "verbatim",
        "04-translation-obfuscation": "paraphrase",
        "05-summary-obfuscation": "summary"
    }
    gt = {}
    for folder, label in type_map.items():
        folder_path = os.path.join(pan_root, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.endswith(".xml"):
                continue
            base = fname[:-4]
            parts = base.split("-source-document")
            if len(parts) != 2:
                continue
            susp = parts[0] + ".txt"
            src  = "source-document" + parts[1] + ".txt"
            gt[(susp, src)] = label
    return gt


_PAIR_NAME_RE = re.compile(r"^(suspicious-document\d+)\.txt--(source-document\d+)\.txt\.json$")

def read_predictions(results_dir: str):
    pred_map = {}
    for root, _, files in os.walk(results_dir):
        for fname in files:
            if not fname.lower().endswith(".json"):
                continue
            if fname.lower() == "evaluation.json":
                continue

            m1 = re.match(r"^(suspicious-document\d+)--(source-document\d+)\.json$", fname)
            m2 = re.match(r"^(suspicious-document\d+)\.txt--(source-document\d+)\.txt\.json$", fname)

            if m1:
                susp = m1.group(1) + ".txt"
                src  = m1.group(2) + ".txt"
            elif m2:
                susp = m2.group(1) + ".txt"
                src  = m2.group(2) + ".txt"
            else:
                continue

            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    segs = json.load(f)
            except Exception as e:
                logging.warning(f"读取失败，跳过: {fpath} ({e})")
                continue

            if not segs:
                pred = "none"
            else:
                cover = {"verbatim": 0, "paraphrase": 0, "summary": 0}
                for s in segs:
                    t = s.get("plagiarism_type", "none")
                    if t == "none":
                        continue
                    a = s.get("source_offset", [0, 0])
                    b = s.get("suspicious_offset", [0, 0])
                    length = max(a[1] - a[0], b[1] - b[0])
                    if t in cover:
                        cover[t] += max(0, int(length))
                pred = max(cover, key=cover.get) if any(cover.values()) else "none"

            pred_map[(susp, src)] = pred

    logging.info(f"[DEBUG] 解析到预测文件数：{len(pred_map)}")
    return pred_map

def align_pairs(gt_map, pred_map):
    keys_gt   = set(gt_map.keys())
    keys_pred = set(pred_map.keys())
    inter     = sorted(keys_gt & keys_pred)
    if not inter:
        logging.error("真值与预测没有交集！请检查：PAN_ROOT / RESULTS_DIR / 文件命名是否一致。")
        return [], [], []

    y_true = [gt_map[k] for k in inter]
    y_pred = [pred_map[k] for k in inter]
    return inter, y_true, y_pred



def plot_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    disp.plot(cmap="Blues", ax=ax, xticks_rotation=30, colorbar=False)
    ax.set_title("Confusion matrix (PAN2013)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)



import numpy as np
import matplotlib.pyplot as plt

def plot_grouped_bars(report_dict, out_path):
    cats = ["none", "verbatim", "paraphrase", "summary"]
    metrics = ["precision", "recall", "f1-score"]
    data = {m: [report_dict.get(c, {}).get(m, 0.0) for c in cats] for m in metrics}

    x = np.arange(len(cats))
    width = 0.25

    colors = [plt.cm.Blues(0.4), plt.cm.Blues(0.6), plt.cm.Blues(0.8)]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - width, data["precision"], width, label="Precision", color=colors[0])
    ax.bar(x,          data["recall"],    width, label="Recall",    color=colors[1])
    ax.bar(x + width,  data["f1-score"],  width, label="F1-score",  color=colors[2])

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Per-class metrics (PAN2013)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_macro_weighted(report_dict, out_path):
    macro = report_dict.get("macro avg", {})
    weight= report_dict.get("weighted avg", {})

    labels = ["Precision", "Recall", "F1-score"]
    macro_vals   = [macro.get("precision",0),  macro.get("recall",0),  macro.get("f1-score",0)]
    weighted_vals= [weight.get("precision",0), weight.get("recall",0), weight.get("f1-score",0)]

    x = np.arange(len(labels))
    width = 0.35

    colors = [plt.cm.Blues(0.5), plt.cm.Blues(0.8)]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - width/2, macro_vals,    width, label="Macro",    color=colors[0])
    ax.bar(x + width/2, weighted_vals, width, label="Weighted", color=colors[1])

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Macro vs Weighted metrics (PAN2013)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    logging.info("读取真值（来自五个类型文件夹的 XML 文件名映射）…")
    gt_map = get_ground_truth_mapping(PAN_ROOT)
    logging.info(f"  真值对数：{len(gt_map)}")

    logging.info("读取预测（来自 pan2013_outputs/*.json）…")
    pred_map = read_predictions(RESULTS_DIR)
    logging.info(f"  预测对数：{len(pred_map)}")

    keys, y_true, y_pred = align_pairs(gt_map, pred_map)
    if not keys:
        return

    aligned_csv = os.path.join(OUT_DIR, "aligned_pairs.csv")
    with open(aligned_csv, "w", encoding="utf-8") as f:
        f.write("susp,src,y_true,y_pred\n")
        for (susp, src), yt, yp in zip(keys, y_true, y_pred):
            f.write(f"{susp},{src},{yt},{yp}\n")
    logging.info(f"[OK] 对齐数据已保存: {aligned_csv}")

    rep = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True)
    rep_path = os.path.join(OUT_DIR, "classification_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    logging.info(f"[OK] 分类报告 JSON: {rep_path}")

    plot_grouped_bars(rep, os.path.join(OUT_DIR, "fig_per_class_metrics.png"))
    plot_confusion(y_true, y_pred, os.path.join(OUT_DIR, "fig_confusion.png"))
    plot_macro_weighted(rep, os.path.join(OUT_DIR, "fig_macro_vs_weighted.png"))
    logging.info(f"[OK] 图像已输出到: {OUT_DIR}")

    common_keys = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    y_true = [gt_map[k] for k in common_keys]
    y_pred = [pred_map[k] for k in common_keys]

    print("\n=== Classification Report (per-class) ===")
    print(classification_report(y_true, y_pred, digits=3))


if __name__ == "__main__":
    main()