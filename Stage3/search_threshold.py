"""
Stage3 验证器阈值搜索脚本
==========================
对 build_evaluate_result.py 输出的含 p_yes 分数的 raw 预测
（如 result/dev_pred_redocred.json），遍历不同阈值 τ，
转官方格式并计算 P / R / F1 / Ign F1，找到 F1 最高的 τ。

τ 语义：P(Yes) >= τ 才认为该候选关系成立。
  - τ 越低 → 保留三元组越多 → recall 高、precision 低；
  - τ 越高 → 保留三元组越少 → precision 高、recall 低。

用法示例（Re-DocRED dev）：
  python search_threshold.py `
      --pred result/dev_pred_redocred.json `
      --rel_info ../datasets/redocred/rel_info.json `
      --data_dir ../datasets/redocred `
      --dataset redocred --split dev

确定最佳 τ 后，用 evaluate.py 以该 τ 生成 test 的官方格式。
"""

import argparse
import json

from evaluate import load_gold, evaluate, load_json, invert_rel_info, to_official_triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="result/dev_pred_redocred.json",
                    help="raw 预测文件（含 p_yes 分数）")
    ap.add_argument("--rel_info", default="../datasets/redocred/rel_info.json")
    ap.add_argument("--data_dir", default="../datasets/redocred")
    ap.add_argument("--dataset", default="redocred", choices=["docred", "redocred"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.93,0.95,0.97,0.99",
                    help="逗号分隔的阈值列表")
    ap.add_argument("--topk", type=int, default=None,
                    help="可选：每实体对只保留 top-k 个关系")
    ap.add_argument("--out", default=None,
                    help="可选：把最佳阈值的官方格式写入该文件")
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    dev_pred = load_json(args.pred)
    name2pid = invert_rel_info(load_json(args.rel_info))
    gold_triples, train_triples = load_gold(args.data_dir, args.split, args.dataset)

    print(f"pred file    : {args.pred}")
    print(f"dataset      : {args.dataset} / {args.split}")
    print(f"gold triples : {len(gold_triples)}")
    print("=" * 72)
    print(f"{'threshold':<12}{'P':>9}{'R':>9}{'F1':>9}{'Ign_F1':>9}{'#pred':>9}")
    print("-" * 72)

    best = {"f1": -1.0, "tau": None, "triples": None, "row": None}

    for tau in thresholds:
        triples, _ = to_official_triples(dev_pred, name2pid, threshold=tau, topk=args.topk)
        m = evaluate(triples, gold_triples, train_triples)
        p, r, f1, ign = m["Precision"], m["Recall"], m["F1"], m["Ign_F1"]
        print(f"{tau:<12}{p * 100:>9.2f}{r * 100:>9.2f}{f1 * 100:>9.2f}{ign * 100:>9.2f}{len(triples):>9}")
        if f1 > best["f1"]:
            best = {"f1": f1, "tau": tau, "triples": triples,
                    "row": (p, r, f1, ign, len(triples))}

    print("-" * 72)
    p, r, f1, ign, n = best["row"]
    print(f"最佳阈值 τ = {best['tau']}  ->  P={p * 100:.2f}  R={r * 100:.2f}  "
          f"F1={f1 * 100:.2f}  Ign_F1={ign * 100:.2f}  #pred={n}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(best["triples"], f, ensure_ascii=False, indent=2)
        print(f"[DONE] 最佳阈值的官方格式已写: {args.out}")

    print("\n提示：确定最佳 τ 后，对 test 用该阈值转官方格式并评估：")
    print(f"  python evaluate.py --pred result/test_pred_redocred.json "
          f"--rel_info ../datasets/redocred/rel_info.json "
          f"--threshold {best['tau']} --out result/test_official.json")


if __name__ == "__main__":
    main()
