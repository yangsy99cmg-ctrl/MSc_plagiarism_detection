# tests/conftest.py
import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import os, io, json, types
import pytest


@pytest.fixture
def tiny_pan(tmp_path):
    root = tmp_path / "pan13-mini"
    (root / "src").mkdir(parents=True)
    (root / "susp").mkdir(parents=True)

    folders = [
        "01-no-plagiarism",
        "02-no-obfuscation",
        "03-random-obfuscation",
        "04-translation-obfuscation",
        "05-summary-obfuscation",
    ]
    for d in folders:
        (root / d).mkdir()

    pairs = [
        ("suspicious-document00001.txt", "source-document00001.txt"),
        ("suspicious-document00002.txt", "source-document00002.txt"),
        ("suspicious-document00003.txt", "source-document00003.txt"),
    ]

    with io.open(root / "pairs", "w", encoding="utf-8") as f:
        for s, r in pairs:
            f.write(f"{s} {r}\n")

    texts = {
        "source-document00001.txt": 'The "quoted sentence" is here. Apples are red.',
        "suspicious-document00001.txt": 'The "quoted sentence" is here. Apples are red.',
        "source-document00002.txt": 'Deep learning advances rapidly in recent years.',
        "suspicious-document00002.txt": 'Recent years have seen rapid advances in deep learning.',
        "source-document00003.txt": "Long paragraph about topic A with many details.",
        "suspicious-document00003.txt": "Short summary of topic A.",
    }
    for name, content in texts.items():
        sub = "src" if name.startswith("source") else "susp"
        with io.open(root / sub / name, "w", encoding="utf-8") as f:
            f.write(content)

    xml_map = {
        "02-no-obfuscation": "suspicious-document00001-source-document00001.xml",
        "04-translation-obfuscation": "suspicious-document00002-source-document00002.xml",
        "05-summary-obfuscation": "suspicious-document00003-source-document00003.xml",
    }
    for d, fname in xml_map.items():
        with io.open(root / d / fname, "w", encoding="utf-8") as f:
            f.write("<dummy/>")

    return root

@pytest.fixture
def out_dir(tmp_path):
    d = tmp_path / "outputs"
    d.mkdir()
    return d