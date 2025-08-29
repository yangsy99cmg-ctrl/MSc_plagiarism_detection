# do_language_detector.py  (pure-functions only)

import os
import io
import tempfile
from typing import List, Tuple, Dict

from PAN2014_3 import SGSPLAG
from utils import (
    extract_sentence_pairs_from_offsets,
    load_doc,
    save_json,
    longest_common_substring_len,
)
from roberta_paraphrase_classifier import classify_sentence_pairs
from ner_matcher import check_named_entity_match

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

from sentence_transformers import SentenceTransformer, util
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
_EmbedMini = SentenceTransformer("all-MiniLM-L6-v2", device=device)  


# =========================
# AAFP alignment (standard & loose parameters)
# =========================
def run_text_alignment(
    susp_path: str,
    src_path: str,
    outdir: str,
    min_match_words: int = 5, 
):
    """Standard AAFP (AAFP = PAN2014 winning proposal)"""
    sgsplag = SGSPLAG(susp_path, src_path, outdir)
    sgsplag.min_sentlen = 2
    sgsplag.min_match_words = min_match_words
    sgsplag.th1 = 0.33
    sgsplag.th2 = 0.33
    sgsplag.th3 = 0.40
    sgsplag.src_gap = 4
    sgsplag.susp_gap = 4
    sgsplag.src_gap_least = 2
    sgsplag.susp_gap_least = 2
    sgsplag.min_plaglen = 80

    sgsplag.process()
    return sgsplag.detections  # [((src_start,src_end),(susp_start,susp_end))]


def run_text_alignment_loose(
    susp_path: str,
    src_path: str,
    outdir: str,
    min_match_words: int = 3,
):
    """Loose parameters AAFP (Second Pass, used to recall paraphrase/summary)"""
    sgsplag = SGSPLAG(susp_path, src_path, outdir)
    sgsplag.th1 = 0.28
    sgsplag.th2 = 0.28
    sgsplag.th3 = 0.35
    sgsplag.min_sentlen = 1
    sgsplag.min_match_words = min_match_words
    sgsplag.src_gap = 8
    sgsplag.susp_gap = 8
    sgsplag.src_gap_least = 2
    sgsplag.susp_gap_least = 2
    sgsplag.min_plaglen = 60

    sgsplag.process()
    return sgsplag.detections


# =========================
# Semantic callback: BM25 + embedding
# =========================
def _simple_sents(t: str) -> List[str]:
    parts, start = [], 0
    for i, ch in enumerate(t):
        if ch in ".!?。\n":
            seg = t[start:i].strip()
            if seg:
                parts.append(seg)
            start = i + 1
    last = t[start:].strip()
    if last:
        parts.append(last)
    return parts


def bm25_realign_sentences(susp_text: str, src_text: str, min_sim: float = 0.85):
    susp_sents = _simple_sents(susp_text)
    src_sents  = _simple_sents(src_text)
    if not susp_sents or not src_sents:
        return []

    bm25 = BM25Okapi([s.split() for s in src_sents])
    src_embs = _EmbedMini.encode(src_sents, convert_to_tensor=True, batch_size=64, show_progress_bar=False)

    detections = []
    for susp_sent in susp_sents:
        scores = bm25.get_scores(susp_sent.split())
        best_idx = int(max(range(len(scores)), key=lambda j: scores[j]))
        susp_emb = _EmbedMini.encode([susp_sent], convert_to_tensor=True, batch_size=1, show_progress_bar=False)
        sim = util.cos_sim(susp_emb, src_embs[best_idx:best_idx+1]).item()
        if sim >= min_sim:
            susp_start = susp_text.find(susp_sent); susp_end = susp_start + len(susp_sent)
            src_start  = src_text.find(src_sents[best_idx]); src_end = src_start + len(src_sents[best_idx])
            if susp_start >= 0 and src_start >= 0:
                detections.append(((src_start, src_end), (susp_start, susp_end)))
    return detections

def bm25_realign_paragraphs(susp_text: str, src_text: str, min_sim: float = 0.78, top_k: int = 3):
    def to_paragraphs(t: str) -> List[str]:
        paras = [p.strip() for p in t.split("\n\n") if p.strip()]
        return [p for p in paras if len(p) >= 150]

    susp_paras = to_paragraphs(susp_text)
    src_paras  = to_paragraphs(src_text)
    if not susp_paras or not src_paras:
        return []

    bm25 = BM25Okapi([p.split() for p in src_paras])
    src_embs = _EmbedMini.encode(src_paras, convert_to_tensor=True, batch_size=32, show_progress_bar=False)

    detections = []
    for sp in susp_paras:
        scores = bm25.get_scores(sp.split())
        top_idx = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:top_k]
        sp_emb = _EmbedMini.encode([sp], convert_to_tensor=True, batch_size=1, show_progress_bar=False)
        for j in top_idx:
            sim = util.cos_sim(sp_emb, src_embs[j:j+1]).item()
            if sim >= min_sim:
                susp_start = susp_text.find(sp); susp_end = susp_start + len(sp)
                src_start  = src_text.find(src_paras[j]); src_end = src_start + len(src_paras[j])
                if susp_start >= 0 and src_start >= 0:
                    detections.append(((src_start, src_end), (susp_start, susp_end)))
    return detections


def align_with_fallback_paths(
    susp_path: str,
    src_path: str,
    out_dir: str,
    *,
    low_alignment: int = 1,
    min_char_cov: int = 200,
    sent_min_sim: float = 0.85,
    para_min_sim: float = 0.78,
    para_topk: int = 3,
    min_match_words: int = 5,
):

    os.makedirs(out_dir, exist_ok=True)

    detections = run_text_alignment(susp_path, src_path, out_dir, min_match_words=min_match_words)

    total_chars = sum(b[1] - b[0] for _, b in detections)
    if (len(detections) < low_alignment) or (total_chars < min_char_cov):
        det_loose = run_text_alignment_loose(susp_path, src_path, out_dir)

        susp_text = load_doc(susp_path)
        src_text = load_doc(src_path)
        det_sent = bm25_realign_sentences(susp_text, src_text, min_sim=sent_min_sim)
        det_para = bm25_realign_paragraphs(susp_text, src_text, min_sim=para_min_sim, top_k=para_topk)

        merged = set()
        for (a, b) in (detections + det_loose + det_sent + det_para):
            merged.add((a[0], a[1], b[0], b[1]))
        detections = [((x0, x1), (y0, y1)) for (x0, x1, y0, y1) in merged]

    return detections


def align_with_fallback_texts(
    susp_text: str,
    src_text: str,
    out_dir: str,
    *,
    low_alignment: int = 1,
    min_char_cov: int = 200,
    sent_min_sim: float = 0.85,
    para_min_sim: float = 0.78,
    para_topk: int = 3,
    min_match_words: int = 5,
):
    """
    Original text version (for GPT2 evaluation):
    Internally, a temporary file is used to reuse the AAFP file interface. The process is the same as align_with_fallback_paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pan_files_") as tmpdir:
        susp_path = os.path.join(tmpdir, "susp.txt")
        src_path = os.path.join(tmpdir, "src.txt")
        with io.open(susp_path, "w", encoding="utf-8") as f: f.write(susp_text)
        with io.open(src_path,  "w", encoding="utf-8") as f: f.write(src_text)

        return align_with_fallback_paths(
            susp_path, src_path, out_dir,
            low_alignment=low_alignment,
            min_char_cov=min_char_cov,
            sent_min_sim=sent_min_sim,
            para_min_sim=para_min_sim,
            para_topk=para_topk,
            min_match_words=min_match_words,
        )


def classify_plagiarism_type(
    src_doc: str,
    susp_doc: str,
    offsets: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    paraphrase_range=(0.35, 0.8),
    lcs_len_verbatim=40,
    lcs_ratio_verbatim=0.20,
    min_len=80
):

    pairs = extract_sentence_pairs_from_offsets(src_doc, susp_doc, offsets)
    if not pairs:
        return []

    sims = classify_sentence_pairs(pairs)           
    ner_matches = check_named_entity_match(pairs)   # NER Jaccard>=0.5

    results = []
    for ((src_span, susp_span), sim, ner_ok) in zip(offsets, sims, ner_matches):
        src_txt = src_doc[src_span[0]:src_span[1]]
        susp_txt = susp_doc[susp_span[0]:susp_span[1]]

        if len(src_txt) < min_len or len(susp_txt) < min_len:
            continue

        lcs_len = longest_common_substring_len(src_txt, susp_txt)
        lcs_ratio = lcs_len / max(1, min(len(src_txt), len(susp_txt)))

        if (lcs_len >= lcs_len_verbatim) or (lcs_ratio >= lcs_ratio_verbatim):
            label = "verbatim"
        else:
            low, high = paraphrase_range
            if sim >= high:
                label = "paraphrase" if ner_ok else "summary"
            elif sim >= low:
                label = "paraphrase" if ner_ok else "summary"
            else:
                len_ratio = len(susp_txt) / max(1, len(src_txt))
                if (lcs_len < 20) and (len_ratio < 0.7):
                    label = "summary"
                else:
                    label = "none"

        if label == "paraphrase":
            if sim < 0.5 and ner_ok:
                label = "summary"
        elif label == "summary":
            if sim >= 0.75 and lcs_len >= 25:
                label = "paraphrase"

        # Quote protection -> verbatim
        if '"' in src_txt and '"' in susp_txt:
            quotes_src = set(part.strip() for part in src_txt.split('"') if len(part.strip()) >= 10)
            quotes_susp = set(part.strip() for part in susp_txt.split('"') if len(part.strip()) >= 10)
            if quotes_src & quotes_susp:
                label = "verbatim"

        results.append({
            "source_offset": src_span,
            "suspicious_offset": susp_span,
            "plagiarism_type": label,
            "similarity": float(sim),
            "lcs_len": int(lcs_len),
            "lcs_ratio": round(lcs_ratio, 3),
            "ner_match": bool(ner_ok),
        })
    return results


# Encapsulation
def detect_and_classify_paths(
    susp_path: str,
    src_path: str,
    out_dir: str,
    *,
    low_alignment: int = 1,
    min_char_cov: int = 200,
    sent_min_sim: float = 0.85,
    para_min_sim: float = 0.78,
    para_topk: int = 3,
    min_match_words: int = 5,
):
    det = align_with_fallback_paths(
        susp_path, src_path, out_dir,
        low_alignment=low_alignment,
        min_char_cov=min_char_cov,
        sent_min_sim=sent_min_sim,
        para_min_sim=para_min_sim,
        para_topk=para_topk,
        min_match_words=min_match_words,
    )
    src_text, susp_text = load_doc(src_path), load_doc(susp_path)
    cls = classify_plagiarism_type(src_text, susp_text, det)
    return det, cls


def detect_and_classify_texts(
    susp_text: str,
    src_text: str,
    out_dir: str,
    *,
    low_alignment: int = 1,
    min_char_cov: int = 200,
    sent_min_sim: float = 0.85,
    para_min_sim: float = 0.78,
    para_topk: int = 3,
    min_match_words: int = 5,
):
    det = align_with_fallback_texts(
        susp_text, src_text, out_dir,
        low_alignment=low_alignment,
        min_char_cov=min_char_cov,
        sent_min_sim=sent_min_sim,
        para_min_sim=para_min_sim,
        para_topk=para_topk,
        min_match_words=min_match_words,
    )
    cls = classify_plagiarism_type(src_text, susp_text, det)
    return det, cls

def detect_and_classify(
    susp_input: str,
    src_input: str,
    out_dir: str,
    *,
    low_alignment: int = 1,
    min_char_cov: int = 200,
    sent_min_sim: float = 0.85,
    para_min_sim: float = 0.75,   
    para_topk: int = 5,           
    min_match_words: int = 5,     
):

    if os.path.exists(susp_input) and os.path.exists(src_input):
        _, cls = detect_and_classify_paths(
            susp_input, src_input, out_dir,
            low_alignment=low_alignment,
            min_char_cov=min_char_cov,
            sent_min_sim=sent_min_sim,
            para_min_sim=para_min_sim,
            para_topk=para_topk,
            min_match_words=min_match_words,
        )
        return cls
    else:
        _, cls = detect_and_classify_texts(
            susp_input, src_input, out_dir,
            low_alignment=low_alignment,
            min_char_cov=min_char_cov,
            sent_min_sim=sent_min_sim,
            para_min_sim=para_min_sim,
            para_topk=para_topk,
            min_match_words=min_match_words,
        )
        return cls