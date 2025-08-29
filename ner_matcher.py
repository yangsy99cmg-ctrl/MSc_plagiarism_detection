# ner_matcher.py
import spacy
nlp = spacy.load('en_core_web_sm')

def extract_entities(sent):
    doc = nlp(sent)
    return set([ent.text.lower() for ent in doc.ents])

# def check_named_entity_match(sentence_pairs):
#     """
#     Input: List of (src_sent, susp_sent)
#     Output: List of bool — whether their entities match exactly
#     """
#     matches = []
#     for src, susp in sentence_pairs:
#         src_ents = extract_entities(src)
#         susp_ents = extract_entities(susp)
#         matches.append(src_ents == susp_ents)
#     return matches

def check_named_entity_match(sentence_pairs):
    matches = []
    for src, susp in sentence_pairs:
        src_ents = extract_entities(src)
        susp_ents = extract_entities(susp)
        if not src_ents and not susp_ents:
            matches.append(False)
            continue
        inter = len(src_ents & susp_ents)
        uni = len(src_ents | susp_ents) or 1
        jacc = inter / uni
        matches.append(jacc >= 0.5)
    return matches