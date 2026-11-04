import json
from typing import Dict, List, Tuple, Any


def build_full_document_evidence(
    raw_doc: Dict[str, Any],
    h_id: int,
    t_id: int,
) -> str:
    """
    Build full-document evidence with entity markers, without [S#] prefixes
    """
    marked_sents = []

    for sid, tokens in enumerate(raw_doc["sents"]):
        spans = []
        spans += _collect_spans_for_sentence(raw_doc, h_id, "H", sid)
        spans += _collect_spans_for_sentence(raw_doc, t_id, "T", sid)

        marked_tokens = mark_sentence_tokens(tokens, spans)
        sent_text = " ".join(marked_tokens)
        marked_sents.append(sent_text)

    return "\n".join(marked_sents)




# ----------------------------
# Entity marking in sentences
# ----------------------------

def _collect_spans_for_sentence(raw_doc: Dict[str, Any], entity_id: int, label: str, sent_id: int) -> List[Tuple[int, int, str]]:
    spans = []
    for m in raw_doc["vertexSet"][entity_id]:
        if int(m.get("sent_id", -1)) == sent_id and "pos" in m:
            start, end = m["pos"]
            spans.append((int(start), int(end), label))
    return spans


def _merge_and_filter_overlaps(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """
    Simple overlap policy:
    - Sort by start asc, longer first
    - Greedily keep non-overlapping spans, skip overlaps
    """
    spans = [s for s in spans if s[0] >= 0 and s[1] > s[0]]
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    kept = []
    last_end = -1
    for s, e, lab in spans:
        if s >= last_end:
            kept.append((s, e, lab))
            last_end = e
    return kept


def mark_sentence_tokens(tokens: List[str], spans: List[Tuple[int, int, str]]) -> List[str]:
    spans = _merge_and_filter_overlaps(spans)
    if not spans:
        return tokens[:]

    out = []
    i = 0
    span_i = 0
    while i < len(tokens):
        if span_i < len(spans) and i == spans[span_i][0]:
            s, e, lab = spans[span_i]
            out.append(f"<{lab}>")
            out.extend(tokens[s:e])
            out.append(f"</{lab}>")
            i = e
            span_i += 1
        else:
            out.append(tokens[i])
            i += 1
    return out


# ----------------------------
# Adapter main
# ----------------------------

def adapt_step15_for_stage2(
    raw_docs: List[Dict[str, Any]],
    step15_docs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    raw_docs: 原始 Re-DocRED json list
    step15_docs: 你的 Stage1.5 输出 list (title/sentences/pairs)
    证据使用整篇文档（不做证据句选择）。
    """
    raw_by_title = {d["title"]: d for d in raw_docs}

    out_docs = []
    missing_titles = 0

    for d in step15_docs:
        title = d["title"]
        raw = raw_by_title.get(title)
        if raw is None:
            missing_titles += 1
            # 保留原样，但标记为 missing
            dd = dict(d)
            dd["_adapter_warning"] = "missing_raw_doc_by_title"
            out_docs.append(dd)
            continue

        dd = {
            "title": title,
            "sentences": d.get("sentences", []),  # 你现有的字符串句子（保留）
            "pairs": []
        }

        for p in d.get("pairs", []):
            h_id = int(p["h_id"])
            t_id = int(p["t_id"])

            # 证据：整篇文档（不做证据句选择）
            marked_text = build_full_document_evidence(
                raw_doc=raw,
                h_id=h_id,
                t_id=t_id,
            )

            pp = dict(p)
            pp["evidence_text_marked"] = marked_text
            dd["pairs"].append(pp)

        out_docs.append(dd)

    if missing_titles > 0:
        print(f"[WARN] {missing_titles} titles in step2 not found in raw docs by title.")

    return out_docs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../datasets/docred/test.json", help="path to original Re-DocRED json (with sents/vertexSet)")
    ap.add_argument("--step15", default="../Stage2/result/Stage2_result_test_docred.json", help="path to your step2 json")
    ap.add_argument("--out", default="input-files/Stage3_input_test_docred.json", help="output path for adapted json")

    args = ap.parse_args()

    with open(args.raw, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)
    with open(args.step15, "r", encoding="utf-8") as f:
        step15_docs = json.load(f)

    adapted = adapt_step15_for_stage2(
        raw_docs=raw_docs,
        step15_docs=step15_docs,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(adapted, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved adapted stage2 input to: {args.out}")


if __name__ == "__main__":
    main()
