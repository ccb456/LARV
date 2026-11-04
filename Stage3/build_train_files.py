import json
from collections import defaultdict
from typing import List, Dict, Tuple
import random

# 注：已删除证据句选择（extract_evidence_sentence_ids），改用整篇文档作为证据。



def load_redocred(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_rel_info(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        rel_map = json.load(f)
    return rel_map


def build_type_constraint_map(docs: List[Dict]) -> Dict[Tuple[str, str], set]:
    """
    (type_h, type_t) -> set(relations)
    """
    type_map = defaultdict(set)

    for doc in docs:
        vertex_set = doc["vertexSet"]
        for label in doc.get("labels", []):
            h = label["h"]
            t = label["t"]
            r = label["r"]

            type_h = vertex_set[h][0]["type"]
            type_t = vertex_set[t][0]["type"]

            type_map[(type_h, type_t)].add(r)

    return type_map

def collect_entity_spans(vertex_set, ent_id, sent_id):
    spans = []
    for m in vertex_set[ent_id]:
        if m["sent_id"] == sent_id:
            spans.append((m["pos"][0], m["pos"][1]))
    return spans


def mark_sentence(tokens, h_spans, t_spans):
    marks = []
    for s, e in h_spans:
        marks.append((s, e, "H"))
    for s, e in t_spans:
        marks.append((s, e, "T"))

    # 按 start 排序，长 span 优先
    marks.sort(key=lambda x: (x[0], -(x[1]-x[0])))

    result = []
    idx = 0
    for s, e, tag in marks:
        if s < idx:
            continue
        result.extend(tokens[idx:s])
        result.append(f"<{tag}>")
        result.extend(tokens[s:e])
        result.append(f"</{tag}>")
        idx = e
    result.extend(tokens[idx:])
    return " ".join(result)


def build_evidence_text(doc, h_id, t_id, evidence_sent_ids):
    sentences = doc["sents"]
    vertex_set = doc["vertexSet"]

    marked_sents = []
    for sid in evidence_sent_ids:
        tokens = sentences[sid]

        h_spans = collect_entity_spans(vertex_set, h_id, sid)
        t_spans = collect_entity_spans(vertex_set, t_id, sid)

        sent = mark_sentence(tokens, h_spans, t_spans)
        marked_sents.append(sent)

    return "\n".join(marked_sents)

def get_full_document_ids(doc: Dict) -> List[int]:
    """直接使用整篇文档的所有句子 id（不再做证据句选择）。"""
    return list(range(len(doc["sents"])))


def build_na_samples(
    doc: Dict,
    pair2rels: Dict,
    type_map: Dict[Tuple[str, str], set],
    rel_map: Dict[str, str],
    n_na: int,
) -> List[Dict]:
    """
    P0：采样「无关系」实体对，生成纯 No 样本，
    让 LLM 学会「一个实体对可以完全没有关系」。
    证据使用整篇文档。
    """
    vertex_set = doc["vertexSet"]
    title = doc["title"]
    n_ent = len(vertex_set)

    all_pairs = [(i, j) for i in range(n_ent) for j in range(n_ent) if i != j]
    na_pairs = [p for p in all_pairs if p not in pair2rels]
    random.shuffle(na_pairs)
    na_pairs = na_pairs[:n_na]

    samples = []
    for h_id, t_id in na_pairs:
        type_h = vertex_set[h_id][0]["type"]
        type_t = vertex_set[t_id][0]["type"]
        candidates = list(type_map.get((type_h, type_t), []))
        if not candidates:
            continue

        r = random.choice(candidates)
        if r not in rel_map:
            continue

        evidence_text = build_evidence_text(doc, h_id, t_id, get_full_document_ids(doc))

        samples.append({
            "title": title,
            "head": vertex_set[h_id][0]["name"],
            "tail": vertex_set[t_id][0]["name"],
            "relation": rel_map[r],
            "evidence_text": evidence_text,
            "label": "No",
        })
    return samples


def build_stage2_samples(
    docs: List[Dict],
    type_map: Dict[Tuple[str, str], set],
    rel_map: Dict[str, str],
    max_no_per_yes: int = 3,
    max_pairs_per_doc: int = 30,
    na_pairs_per_doc: int = 5,           # P0：NA 负样本
) -> List[Dict]:

    samples = []

    for doc in docs:
        title = doc["title"]
        vertex_set = doc["vertexSet"]
        labels = doc.get("labels", [])

        pair2rels = defaultdict(list)

        for lab in labels:
            pair = (lab["h"], lab["t"])
            pair2rels[pair].append(lab["r"])

        used_pairs = list(pair2rels.keys())
        random.shuffle(used_pairs)
        used_pairs = used_pairs[:max_pairs_per_doc]

        for (h_id, t_id) in used_pairs:
            gold_rels = set(pair2rels[(h_id, t_id)])

            # 证据：整篇文档（不做证据句选择）
            evidence_text = build_evidence_text(doc, h_id, t_id, get_full_document_ids(doc))

            head_name = vertex_set[h_id][0]["name"]
            tail_name = vertex_set[t_id][0]["name"]
            type_h = vertex_set[h_id][0]["type"]
            type_t = vertex_set[t_id][0]["type"]

            candidates = list(type_map.get((type_h, type_t), []))
            neg_rels = [r for r in candidates if r not in gold_rels]
            random.shuffle(neg_rels)
            neg_rels = neg_rels[:max_no_per_yes * len(gold_rels)]

            # -------- Yes --------
            for r in gold_rels:
                if r not in rel_map:
                    continue
                samples.append({
                    "title": title, "head": head_name, "tail": tail_name,
                    "relation": rel_map[r], "evidence_text": evidence_text,
                    "label": "Yes",
                })

            # -------- No（类型合法但关系错） --------
            for r in neg_rels:
                if r not in rel_map:
                    continue
                samples.append({
                    "title": title, "head": head_name, "tail": tail_name,
                    "relation": rel_map[r], "evidence_text": evidence_text,
                    "label": "No",
                })

        # -------- P0：NA 样本（无关系 pair） --------
        if na_pairs_per_doc > 0:
            samples.extend(build_na_samples(
                doc, pair2rels, type_map, rel_map, na_pairs_per_doc
            ))

    return samples


def save_jsonl(samples: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    from collections import Counter

    train_path = "../datasets/redocred/train_revised.json"
    rel_info_path = "../datasets/redocred/rel_info.json"
    out_path = "stage3_lora_train.jsonl"

    docs = load_redocred(train_path)
    type_map = build_type_constraint_map(docs)
    rel_map = load_rel_info(rel_info_path)

    samples = build_stage2_samples(
        docs,
        type_map,
        rel_map,
        max_no_per_yes=2,            # 每个正例配几个"类型合法但关系错"的负例
        max_pairs_per_doc=20,        # 每文档最多取多少正例 pair
        na_pairs_per_doc=5,          # P0
    )

    save_jsonl(samples, out_path)
    print(f"Total samples: {len(samples)}")
    print(f"Saved to: {out_path}")

    # -------- 统计 Yes/No 比例 --------
    counter = Counter()
    for s in samples:
        counter[s["label"]] += 1
    total = sum(counter.values())
    print("\nLabel distribution:")
    for k, v in counter.items():
        print(f"  {k}: {v}  ({v / total:.3f})")

    # -------- 证据长度统计 --------
    lengths = sorted(len(s["evidence_text"].split()) for s in samples)
    n = len(lengths)
    print("\nEvidence length (approx tokens):")
    print(f"  min={lengths[0]}, max={lengths[-1]}, avg={sum(lengths) / n:.1f}")
    print(f"  P90={lengths[int(0.90 * n)]}, P95={lengths[int(0.95 * n)]}, P99={lengths[int(0.99 * n)]}")

