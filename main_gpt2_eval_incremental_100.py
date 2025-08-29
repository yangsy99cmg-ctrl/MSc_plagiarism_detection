# main_gpt2_eval_es_top3.py  —— Incremental run 100 items/version
import os
import re
import time
import json
from glob import glob
from collections import defaultdict, Counter

import pandas as pd
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import matplotlib.pyplot as plt

TRAIN_PATH = r"C:/Users/YangSY/Desktop/homework/MSc_desk/owt_5_samples/owt_sample_random5pct.parquet"
# TRAIN_PATH = r"C:/Users/YangSY/Desktop/homework/MSc_desk/openwebtext_sample.parquet"
GEN_PATH   = r"C:/Users/YangSY/Desktop/homework/MSc_desk/GPT_generated_text/pretrainedGPT/large-762M-k40.train.txt"

ES_URL        = "http://localhost:9200"
# ES_INDEX_NAME = "gpt2_train_index_top3"
ES_INDEX_NAME = "gpt2_train_index_top3_5pct"
# OUTPUT_DIR    = r"D:/yangsy/results/gpt2_5pct_top3"
OUTPUT_DIR    = r"D:/yangsy/results/gpt2_5pct_top3_new"

BATCH_PER_RUN = 100
TOP_K         = 3   # Retrieve the top 3 stories for each story

# ES search slightly tightened
MIN_SHOULD_MATCH = "75%"

MIN_STORY_KEEP     = 200 
TARGET_STORY_RANGE = (2500, 4000)
LONG_STORY_THRESH  = 8000

VALID_TYPES            = {"verbatim", "paraphrase", "summary"}
MIN_SEG_LEN_DEFAULT    = 120
MIN_SEG_LEN_FOR_SHORTS = 150

from do_language_detector import detect_and_classify_texts

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def split_by_blanklines(text: str):
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

def merge_to_target(segments, lo=TARGET_STORY_RANGE[0], hi=TARGET_STORY_RANGE[1]):
    """Merge adjacent stories and try to adjust each story to the [lo,hi] interval; the last story should not be too short"""
    merged, buf, blen = [], [], 0
    for seg in segments:
        L = len(seg)
        if blen + L <= hi:
            buf.append(seg); blen += L
            if blen >= lo:
                merged.append("\n\n".join(buf)); buf=[]; blen=0
        else:
            if buf:
                merged.append("\n\n".join(buf)); buf=[]; blen=0
            if lo <= L <= hi:
                merged.append(seg)
            else:
                buf=[seg]; blen=L
    if buf:
        if merged and len(merged[-1]) + blen <= hi:
            merged[-1] = merged[-1] + "\n\n" + "\n\n".join(buf)
        else:
            merged.append("\n\n".join(buf))
    return merged

def load_and_make_long_stories(gen_path: str):
    with open(gen_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = clean_text(f.read())
    parts = split_by_blanklines(raw)
    parts = [p for p in parts if len(p) >= MIN_STORY_KEEP]
    stories = merge_to_target(parts, *TARGET_STORY_RANGE)
    return stories

def ensure_index_english(es: Elasticsearch, index_name: str, train_df: pd.DataFrame):
    if es.indices.exists(index=index_name):
        print(f"[OK] Reusing existing indexes：{index_name}")
        return
    print(f"[BUILD] Create an index {index_name} (analyzer=english)")
    es.indices.create(
        index=index_name,
        mappings={"properties": {"text": {"type": "text", "analyzer": "english"}}},
        settings={"index": {"number_of_shards": 1, "number_of_replicas": 0}}
    )
    df = train_df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 100]  # Filtering very short training texts
    print(f"[LOAD] Training Documents:{len(df)}")

    def gen_actions(df_in):
        for i, row in df_in.iterrows():
            yield {"_index": index_name, "_id": int(i), "_source": {"text": row["text"]}}

    bulk(es, gen_actions(df), chunk_size=1000, request_timeout=120)
    print("[OK] Index building completed")

def es_topk_match(es: Elasticsearch, index: str, story: str, k: int = TOP_K):
    res = es.search(
        index=index,
        query={"match": {"text": {"query": story, "minimum_should_match": MIN_SHOULD_MATCH}}},
        size=k
    )
    hits = res.get("hits", {}).get("hits", [])
    return [(h["_id"], h["_source"]["text"], h["_score"]) for h in hits]

def keep_real_segments(segments, story_len):
    if not segments:
        return []
    min_len = MIN_SEG_LEN_FOR_SHORTS if story_len < 1000 else MIN_SEG_LEN_DEFAULT
    kept = []
    for seg in segments:
        t = seg.get("plagiarism_type", "none")
        if t not in VALID_TYPES:
            continue
        a = seg.get("source_offset", [0, 0])
        b = seg.get("suspicious_offset", [0, 0])
        length = max(a[1]-a[0], b[1]-b[0])
        if length >= min_len:
            kept.append(seg)
    return kept

def excerpt(txt: str, span, max_chars=220):
    s, e = span
    s = max(0, s); e = min(len(txt), e)
    frag = txt[s:e]
    return frag if len(frag) <= max_chars else frag[:max_chars] + "…"

# ======= Visualization/Aggregation (Scan and Output) =======
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
            merged.append(list(s))
        else:
            last = merged[-1]
            if iou((last[0], last[1]), (s[0], s[1])) >= iou_th or s[0] <= last[1]:
                last[1] = max(last[1], s[1])
            else:
                merged.append([s[0], s[1]])
    return [(a, b) for a, b in merged]

def aggregate_and_plot(out_dir: str):
    files = sorted(glob(os.path.join(out_dir, "gen_[0-9][0-9][0-9][0-9][0-9][0-9].json")))
    if not files:
        print("[WARN] No detail JSON found, skipping visualization")
        return
    story_n = 0
    story_pos = 0
    coverages = []
    type_coverages = defaultdict(list)
    type_counter = Counter()
    aligned_counts = []

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        story_n += 1
        L = data.get("story_chars", 0) or 1
        spans_all = []
        spans_by_type = defaultdict(list)
        aligned_cnt = 0
        for cand in data.get("candidates", []):
            if cand.get("aligned"):
                aligned_cnt += 1
                for seg in cand.get("segments", []):
                    t = seg.get("type", "none")
                    if t not in VALID_TYPES:
                        continue
                    s = seg.get("suspicious_offset", [0, 0])
                    if s[1] <= s[0]:
                        continue
                    spans_all.append((s[0], s[1]))
                    spans_by_type[t].append((s[0], s[1]))
                    type_counter[t] += 1
        aligned_counts.append(aligned_cnt)

        merged = merge_spans(spans_all, iou_th=0.5) if spans_all else []
        cov = sum(b - a for a, b in merged) / L
        coverages.append(cov)
        if cov > 0:
            story_pos += 1

        for t, lst in spans_by_type.items():
            mm = merge_spans(lst, iou_th=0.5) if lst else []
            type_coverages[t].append(sum(b - a for a, b in mm) / L if L else 0.0)

    # Drawing figures
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure()
    plt.hist(aligned_counts, bins=range(0, TOP_K + 2), align="left", rwidth=0.8)
    plt.xlabel(f"Aligned candidates per story (0–{TOP_K})")
    plt.ylabel("Count of stories")
    plt.title("Aligned candidates distribution")
    plt.savefig(os.path.join(plots_dir, "aligned_candidates_hist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.hist([min(1.0, c) for c in coverages], bins=20)
    plt.xlabel("Story-level coverage (merged suspicious spans)")
    plt.ylabel("Count of stories")
    plt.title("Coverage distribution")
    plt.savefig(os.path.join(plots_dir, "coverage_hist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    labels = list(VALID_TYPES)
    sizes = [type_counter[t] for t in labels]
    if sum(sizes) == 0:
        sizes = [1, 1, 1]
    plt.figure()
    plt.pie(sizes, labels=labels, autopct="%.1f%%")
    plt.title("Segment type proportion (by count)")
    plt.savefig(os.path.join(plots_dir, "type_pie.png"), dpi=150, bbox_inches="tight")
    plt.close()

    aggregate = {
        "stories_total": story_n,
        "stories_positive": story_pos,
        "story_level_rate": (story_pos / story_n) if story_n else 0.0,
        "mean_coverage": (sum(coverages) / len(coverages)) if coverages else 0.0,
        "mean_coverage_by_type": {t: (sum(v)/len(v) if v else 0.0) for t, v in type_coverages.items()},
        "segment_count_by_type": dict(type_counter),
        "aligned_candidates_hist": Counter(aligned_counts)
    }
    with open(os.path.join(out_dir, "aggregate.json"), "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    print(f"[OK] Visualization saved to:{plots_dir}")
    print(f"[OK] Aggregate statistics:{os.path.join(out_dir, 'aggregate.json')}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    es = Elasticsearch(ES_URL)
    train_df = pd.read_parquet(TRAIN_PATH)
    ensure_index_english(es, ES_INDEX_NAME, train_df)

    stories = load_and_make_long_stories(GEN_PATH)
    total_stories = len(stories)

    existing = sorted(glob(os.path.join(OUTPUT_DIR, "gen_[0-9][0-9][0-9][0-9][0-9][0-9].json")))
    start_idx = len(existing)
    end_idx   = min(start_idx + BATCH_PER_RUN, total_stories)

    if start_idx >= total_stories:
        print(f"[DONE] All stories have been processed.(Totally {total_stories} stories) Only visualizations will be updated.")
        aggregate_and_plot(OUTPUT_DIR)
        return

    print(f"[RUN] This round of increments: processing stories {start_idx}..{end_idx-1}(All {end_idx-start_idx} stories / Total {total_stories})")
    print(f"[OUT] Output directory:{OUTPUT_DIR}")

    for si in tqdm(range(start_idx, end_idx), desc="Processing Stories"):
        story = stories[si]
        story_len = len(story)

        sum_path = os.path.join(OUTPUT_DIR, f"gen_{si:06d}__summary.json")
        out_path = os.path.join(OUTPUT_DIR, f"gen_{si:06d}.json")
        if os.path.exists(sum_path) and os.path.exists(out_path):
            continue

        t0 = time.time()
        try:
            cands = es_topk_match(es, ES_INDEX_NAME, story, TOP_K)
        except Exception as e:
            print(f"[STORY {si+1}/{total_stories}] ES 检索失败：{e}")
            cands = []
        t1 = time.time()

        align_times = []
        results = []
        aligned_cnt = 0

        for (doc_id, doc_text, es_score) in cands:
            a0 = time.time()
            try:
                det_offsets, det_segments = detect_and_classify_texts(story, doc_text, OUTPUT_DIR)
            except Exception as e:
                det_offsets, det_segments = [], []
                print(f"[WARN] 对齐异常 (story={si}, doc_id={doc_id}): {e}")

            kept = keep_real_segments(det_segments, story_len)
            aligned = bool(kept)
            if aligned:
                aligned_cnt += 1
            a1 = time.time()
            align_times.append(round(a1 - a0, 2))

            rec = {
                "doc_id": str(doc_id),
                "es_score": float(es_score),
                "aligned": aligned,
                "segments": []
            }
            for seg in kept:
                s_span = seg.get("suspicious_offset")
                d_span = seg.get("source_offset")
                rec["segments"].append({
                    "type": seg.get("plagiarism_type"),
                    "similarity": seg.get("similarity"),
                    "lcs_len": seg.get("lcs_len"),
                    "lcs_ratio": seg.get("lcs_ratio"),
                    "ner_match": seg.get("ner_match"),
                    "suspicious_offset": s_span,
                    "source_offset": d_span,
                    "susp_excerpt": excerpt(story, s_span, 220),
                    "src_excerpt": excerpt(doc_text, d_span, 220),
                })
            results.append(rec)

        t2 = time.time()

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "story_index": si,
                    "story_chars": story_len,
                    "candidates": results
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to write details:{e}")

        try:
            with open(sum_path, "w", encoding="utf-8") as f:
                json.dump({
                    "story_index": si,
                    "story_chars": story_len,
                    "es_time_sec": round(t1 - t0, 3),
                    "align_times_sec": align_times,
                    "es_candidates": len(cands),
                    "aligned_candidates": aligned_cnt
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to write summary:{e}")

        print(f"[STORY {si+1}/{total_stories}] {round(t2-t0,1)}s | len={story_len} | "
              f"ES={round(t1-t0,2)}s | CAND={len(cands)} | aligned={aligned_cnt} | "
              f"align_each={align_times}")

    aggregate_and_plot(OUTPUT_DIR)
    print(f"[DONE] This round of increments is complete. See:{OUTPUT_DIR}")

if __name__ == "__main__":
    main()