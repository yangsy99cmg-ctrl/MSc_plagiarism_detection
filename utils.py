# utils.py
import json
import re

def extract_sentence_pairs_from_offsets(src_text, susp_text, offset_pairs):
    """
    Input: offset_pairs = [((src_start, src_end), (susp_start, susp_end))]
    Output: list of (src_sent, susp_sent)
    """
    pairs = []
    for src_range, susp_range in offset_pairs:
        src_sent = src_text[src_range[0]:src_range[1]]
        susp_sent = susp_text[susp_range[0]:susp_range[1]]
        pairs.append((src_sent, susp_sent))
    return pairs

def load_doc(path):
    with open(path, encoding='utf-8') as f:
        return f.read()

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def longest_common_substring_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    best = 0
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev + 1
                if dp[j] > best:
                    best = dp[j]
            else:
                dp[j] = 0
            prev = tmp
    return best

# QUOTE_RE = re.compile(r'(^|\s)[\'"](.*?)[\'"](\s|$)')
QUOTE_RE = re.compile(r'([\'"“”‘’]).{20,}\1', re.DOTALL)


def has_proper_quotes(text: str) -> bool:

    return bool(QUOTE_RE.search(text))