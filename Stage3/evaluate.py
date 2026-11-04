"""
评估工具：raw 预测 -> 官方三元组 -> P/R/F1/Ign_F1
=================================================
整合原 convert_pred_to_official.py（转官方格式）与 evaluate.py（算指标）。

两种用法：
  1) 只转官方格式（不评估）：
     python evaluate.py --pred result/dev_pred_redocred.json \
         --rel_info ../datasets/redocred/rel_info.json \
         --threshold 0.97 --out result/dev_official.json

  2) 转格式 + 评估（推荐）：
     python evaluate.py --pred result/dev_pred_redocred.json \
         --rel_info ../datasets/redocred/rel_info.json \
         --threshold 0.97 \
         --data_dir ../datasets/redocred --dataset redocred --split dev \
         --out result/dev_official.json
"""

import argparse
import json


# ===============================
# 通用工具
# ===============================
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def invert_rel_info(rel_info):
    """{pid: name} -> {name: pid}"""
    name2pid = {}
    for pid, name in rel_info.items():
        if name in name2pid and name2pid[name] != pid:
            print(f"[WARN] duplicate relation name: {name} -> {name2pid[name]} / {pid}")
            continue
        name2pid[name] = pid
    return name2pid


# ===============================
# 转官方格式
# ===============================
def to_official_triples(dev_pred, name2pid, threshold=None, topk=None):
    """
    把含 p_yes 的 raw 预测按阈值/topk 转成官方三元组格式。
    返回 (triples, missing_rel)，missing_rel 为无法映射到关系 ID 的条数。
    """
    out = []
    seen = set()
    missing_rel = 0

    for doc in dev_pred:
        title = doc["title"]
        for pair in doc.get("relations", []):
            h_id = pair["h_id"]
            t_id = pair["t_id"]
            rel_list = pair.get("relations", [])

            # 含 p_yes 时：按分数排序，再按阈值/topk 过滤
            if rel_list and isinstance(rel_list[0], dict) and "p_yes" in rel_list[0]:
                rel_list = sorted(rel_list, key=lambda x: x.get("p_yes", 0.0), reverse=True)
                if topk is not None:
                    rel_list = rel_list[:topk]
                if threshold is not None:
                    rel_list = [x for x in rel_list if x.get("p_yes", 0.0) >= threshold]

            for r_item in rel_list:
                rel_name = r_item["relation"] if isinstance(r_item, dict) else str(r_item)
                pid = name2pid.get(rel_name)
                if pid is None:
                    missing_rel += 1
                    continue
                key = (title, h_id, t_id, pid)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"title": title, "h_idx": h_id, "t_idx": t_id, "r": pid})

    return out, missing_rel


# ===============================
# 评估
# ===============================
def load_gold(data_dir, split="test", dataset="docred"):
    """返回 (gold_triples, train_triples)，train 用于 Ign F1。"""
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    if dataset == "redocred":
        train_file = "train_revised.json"
        eval_file = f"{split}_revised.json"
    else:  # docred
        train_file = "train_annotated.json"
        eval_file = f"{split}.json"

    train_data = load(f"{data_dir}/{train_file}")
    eval_data = load(f"{data_dir}/{eval_file}")

    train_triples = set()
    for doc in train_data:
        title = doc["title"]
        for label in doc.get("labels", []):
            train_triples.add((title, label["h"], label["t"], label["r"]))

    gold_triples = set()
    for doc in eval_data:
        title = doc["title"]
        for label in doc.get("labels", []):
            gold_triples.add((title, label["h"], label["t"], label["r"]))

    return gold_triples, train_triples


def evaluate(preds, gold_triples, train_triples):
    """preds: list of {title, h_idx, t_idx, r}，返回 P/R/F1/Ign_F1 字典。"""
    pred_triples = set((p["title"], p["h_idx"], p["t_idx"], p["r"]) for p in preds)

    # ===== Standard Micro P/R/F1 =====
    tp = len(pred_triples & gold_triples)
    fp = len(pred_triples - gold_triples)
    fn = len(gold_triples - pred_triples)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    # ===== Ign F1（忽略训练集已见三元组）=====
    ign_gold = gold_triples - train_triples
    ign_pred = pred_triples - train_triples
    ign_tp = len(ign_pred & ign_gold)
    ign_fp = len(ign_pred - ign_gold)
    ign_fn = len(ign_gold - ign_pred)
    ign_precision = ign_tp / (ign_tp + ign_fp) if ign_tp + ign_fp > 0 else 0.0
    ign_recall = ign_tp / (ign_tp + ign_fn) if ign_tp + ign_fn > 0 else 0.0
    ign_f1 = 2 * ign_precision * ign_recall / (ign_precision + ign_recall) if ign_precision + ign_recall > 0 else 0.0

    return {"Precision": precision, "Recall": recall, "F1": f1, "Ign_F1": ign_f1}


def main():
    ap = argparse.ArgumentParser()

    # ---- 转格式 ----
    ap.add_argument("--pred", default="result/dev_pred_redocred.json",
                    help="raw 预测文件（含 p_yes 分数）")
    ap.add_argument("--rel_info", default="../datasets/redocred/rel_info.json",
                    help="关系映射 {pid: name}")
    ap.add_argument("--out", default=None,
                    help="官方格式输出文件（可选）")
    ap.add_argument("--threshold", type=float, default=0.97,
                    help="p_yes 阈值（dev 搜索出的最佳值 0.97）")
    ap.add_argument("--topk", type=int, default=None,
                    help="每实体对只保留 top-k 个关系（可选）")

    # ---- 评估（可选） ----
    ap.add_argument("--data_dir", default=None,
                    help="数据集目录；提供则同时计算 P/R/F1/Ign_F1")
    ap.add_argument("--dataset", default="redocred", choices=["docred", "redocred"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    args = ap.parse_args()

    dev_pred = load_json(args.pred)
    name2pid = invert_rel_info(load_json(args.rel_info))

    triples, missing = to_official_triples(dev_pred, name2pid,
                                           threshold=args.threshold, topk=args.topk)
    if missing > 0:
        print(f"[WARN] {missing} relations not found in rel_info (skipped).")
    print(f"[OK] total triples: {len(triples)}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(triples, f, ensure_ascii=False, indent=2)
        print(f"[DONE] wrote {args.out}")

    if args.data_dir:
        gold_triples, train_triples = load_gold(args.data_dir, args.split, args.dataset)
        print(f"gold_triples: {len(gold_triples)}")
        print(f"train_triples: {len(train_triples)}")
        metrics = evaluate(triples, gold_triples, train_triples)
        print("=" * 60)
        print(f"[{args.dataset} Evaluation] split={args.split}  threshold={args.threshold}")
        for k, v in metrics.items():
            print(f"{k:10s}: {v * 100:.2f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
