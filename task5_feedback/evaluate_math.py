"""Task 5 Steps 1 and 3 - in-domain (GSM8K) and out-of-domain (SVAMP) evaluation.

Phase 1 (generation): SFT, RLVR and RLAIF each answer every problem with identical
deterministic greedy decoding (cap = math_max_new_tokens), in the fixed file order.
Phase 2 (scoring):
  * exact verifier (released exact_reward): accuracy; format compliance = a designated
    `#### <number>` final answer was parsed; response length; failure-type breakdown;
  * fixed pairwise AI judge (released PairwiseAIJudge, orientation-balanced, cached):
    RLVR vs SFT and RLAIF vs SFT (and RLVR vs RLAIF). Win = 1, tie = 0.5, loss = 0;
  * verifier-judge agreement on every compared pair: the verifier "prefers" the response
    with the higher exact reward (tie when equal). Reported overall, on verifier-decisive
    pairs (exactly one correct), and as Cohen's kappa.

Outputs: results/task5_feedback/<dataset>/{generations_<policy>.jsonl, pairwise_<a>_vs_<b>.jsonl,
summary.json}. The judge cache is shared: results/task5_feedback/judge_cache.json.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.eval_utils import length_stats
from common.generation import batch_generate
from common.logging_utils import save_json
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final, numerically_equal

POLICIES = ["sft", "rlvr", "rlaif"]
COMPARISONS = [("rlvr", "sft"), ("rlaif", "sft"), ("rlvr", "rlaif")]


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


_NUM_RE = __import__("re").compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def failure_type(response: str, gold: str, truncated: bool) -> str:
    pred = extract_designated_final(response)
    if pred is not None and numerically_equal(pred, gold):
        return "correct"
    if pred is None:
        return "no_final_truncated" if truncated else "no_final_format"
    nums = [n.replace(",", "") for n in _NUM_RE.findall(response)]
    if any(numerically_equal(n, gold) for n in nums):
        return "wrong_final_gold_mentioned"  # gold appeared as an intermediate/discarded number
    return "wrong_final"


def generate_all(cfg, rows, tok, dataset, outdir, batch_size, policies, force=False):
    for name in policies:
        out = outdir / f"generations_{name}.jsonl"
        if out.exists() and not force:
            print(f"[task5:{dataset}] {out.name} exists; reusing")
            continue
        model = load_frozen_policy(cfg, name)
        recs = []
        t0 = time.perf_counter()
        for s in range(0, len(rows), batch_size):
            chunk = rows[s:s + batch_size]
            gen = batch_generate(model, tok, [prompt_messages(r) for r in chunk], max_prompt_length=512,
                                 max_new_tokens=int(cfg["math_max_new_tokens"]), do_sample=False)
            for r, txt, n, tr in zip(chunk, gen["responses"], gen["response_lengths"], gen["truncated"]):
                gold = str(r["gold_final"])
                recs.append({"id": str(r.get("source_index", r.get("prompt_id"))), "policy": name,
                             "question": r["question"], "gold_final": gold, "response": txt,
                             "response_tokens": int(n), "truncated": bool(tr),
                             "pred_final": extract_designated_final(txt), "exact": exact_reward(txt, gold),
                             "format_ok": extract_designated_final(txt) is not None,
                             "failure_type": failure_type(txt, gold, bool(tr))})
        write_jsonl(out, recs)
        acc = np.mean([r["exact"] for r in recs])
        print(f"[task5:{dataset}] {name}: acc {acc:.3f} ({time.perf_counter() - t0:.0f}s)")
        model = None
        clear_gpu()


def pairwise(judge, gens, a, b, outdir):
    A = {r["id"]: r for r in gens[a]}
    B = {r["id"]: r for r in gens[b]}
    recs = []
    for pid, ra in A.items():
        rb = B[pid]
        t0 = time.perf_counter()
        pref = judge.compare(ra["question"], ra["response"], rb["response"])
        score = {"A": 1.0, "B": 0.0, "TIE": 0.5}[pref]
        v = "A" if ra["exact"] > rb["exact"] else "B" if rb["exact"] > ra["exact"] else "TIE"
        recs.append({"id": pid, "a": a, "b": b, "judge": pref, "score_a": score, "verifier": v,
                     "exact_a": ra["exact"], "exact_b": rb["exact"], "judge_seconds": time.perf_counter() - t0})
    write_jsonl(outdir / f"pairwise_{a}_vs_{b}.jsonl", recs)
    return recs


def agreement_stats(recs):
    from sklearn.metrics import cohen_kappa_score

    j = [r["judge"] for r in recs]
    v = [r["verifier"] for r in recs]
    dec = [r for r in recs if r["verifier"] != "TIE"]
    both_dec = [r for r in dec if r["judge"] != "TIE"]
    return {
        "n_pairs": len(recs),
        "win_rate": float(np.mean([r["score_a"] for r in recs])),
        "wins": sum(r["judge"] == "A" for r in recs), "losses": sum(r["judge"] == "B" for r in recs),
        "ties": sum(r["judge"] == "TIE" for r in recs),
        "agreement_all": float(np.mean([x == y for x, y in zip(j, v)])),
        "kappa": float(cohen_kappa_score(v, j, labels=["A", "B", "TIE"])) if len(set(v) | set(j)) > 1 else float("nan"),
        "verifier_decisive_pairs": len(dec),
        "judge_agrees_on_decisive": float(np.mean([r["judge"] == r["verifier"] for r in dec])) if dec else float("nan"),
        "judge_tie_on_decisive": float(np.mean([r["judge"] == "TIE" for r in dec])) if dec else float("nan"),
        "judge_contradicts_on_decisive": float(np.mean([r["judge"] not in (r["verifier"], "TIE") for r in dec])) if dec else float("nan"),
        "judge_acc_on_decisive_excluding_ties": float(np.mean([r["judge"] == r["verifier"] for r in both_dec])) if both_dec else float("nan"),
        "verifier_tie_pairs": len(recs) - len(dec),
        "judge_decisive_on_verifier_ties": float(np.mean([r["judge"] != "TIE" for r in recs if r["verifier"] == "TIE"]))
        if len(recs) > len(dec) else float("nan"),
        "verifier_tie_judge_pref_a": float(np.mean([r["judge"] == "A" for r in recs if r["verifier"] == "TIE"]))
        if len(recs) > len(dec) else float("nan"),
        "mean_judge_seconds": float(np.mean([r["judge_seconds"] for r in recs])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--limit", type=int, help="optional fixed prefix for a quick run")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg, rows, tok = load_math_evaluation(args.config, args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    print("Rows:", len(rows), "| Policies:", list(policy_specs(cfg)))
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback" / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)

    generate_all(cfg, rows, tok, args.dataset, outdir, args.batch_size, POLICIES, force=args.force)
    gens = {p: read_jsonl(outdir / f"generations_{p}.jsonl") for p in POLICIES}

    summary = {"dataset": args.dataset, "n": len(rows), "decoding": "greedy",
               "max_new_tokens": int(cfg["math_max_new_tokens"]), "policies": {}}
    for p in POLICIES:
        g = gens[p]
        ft = {}
        for r in g:
            ft[r["failure_type"]] = ft.get(r["failure_type"], 0) + 1
        summary["policies"][p] = {
            "exact_accuracy": float(np.mean([r["exact"] for r in g])),
            "format_compliance": float(np.mean([r["format_ok"] for r in g])),
            "truncation_rate": float(np.mean([r["truncated"] for r in g])),
            "length": length_stats([r["response_tokens"] for r in g]),
            "failure_types": ft,
        }

    if not args.skip_judge:
        judge = PairwiseAIJudge(cfg, repo_path(cfg["results_dir"]) / "task5_feedback" / "judge_cache.json")
        summary["pairwise"] = {}
        for a, b in COMPARISONS:
            recs = pairwise(judge, gens, a, b, outdir)
            summary["pairwise"][f"{a}_vs_{b}"] = agreement_stats(recs)
            s = summary["pairwise"][f"{a}_vs_{b}"]
            print(f"[task5:{args.dataset}] {a} vs {b}: win {s['win_rate']:.3f} (W{s['wins']}/T{s['ties']}/L{s['losses']}) "
                  f"verifier-judge agreement {s['agreement_all']:.3f}")
        judge = None
        clear_gpu()

    save_json(outdir / "summary.json", summary)
    for p, s in summary["policies"].items():
        print(f"[task5:{args.dataset}] {p}: acc {s['exact_accuracy']:.3f} format {s['format_compliance']:.3f} "
              f"len {s['length']['mean']:.0f}±{s['length']['std']:.0f}")


if __name__ == "__main__":
    main()
