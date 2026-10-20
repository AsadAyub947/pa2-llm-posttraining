from __future__ import annotations

import argparse
import math
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.eval_utils import grad_norm, peak_memory_gib, reset_peak_memory
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prompt_fits(tokenizer, row, max_length: int) -> bool:
    """encode_prompt_response raises if the prompt alone does not fit; filter such rows once."""
    ids = tokenizer.apply_chat_template(prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True)
    return len(ids) < max_length


def concat_left_padded(a: dict, b: dict, pad_id: int) -> dict:
    """Stack two left-padded batches (chosen, rejected) into one forward pass."""
    width = max(a["input_ids"].shape[1], b["input_ids"].shape[1])

    def lpad(t, value):
        pad = width - t.shape[1]
        if pad == 0:
            return t
        return torch.cat([torch.full((t.shape[0], pad), value, dtype=t.dtype), t], dim=1)

    return {
        "input_ids": torch.cat([lpad(a["input_ids"], pad_id), lpad(b["input_ids"], pad_id)]),
        "attention_mask": torch.cat([lpad(a["attention_mask"], 0), lpad(b["attention_mask"], 0)]),
        "response_mask": torch.cat([lpad(a["response_mask"], 0), lpad(b["response_mask"], 0)]),
    }


def pair_logps(model, chosen: dict, rejected: dict, pad_id: int, device):
    """Summed response-token log-probs for chosen and rejected (teacher forcing)."""
    batch = concat_left_padded(chosen, rejected, pad_id)
    batch = {k: v.to(device) for k, v in batch.items()}
    seq, _, _ = response_sequence_logprobs(model, batch)
    n = chosen["input_ids"].shape[0]
    return seq[:n], seq[n:]


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    n_before = len(rows)
    rows = [r for r in rows if prompt_fits(tokenizer, r, int(cfg["max_sequence_length"]))]
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "n_filtered_long_prompts": n_before - len(rows),
        "dataset_path": str(path),
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None,
                 beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    model, tok, loader, opt = bundle["model"], bundle["tokenizer"], bundle["loader"], bundle["optimizer"]
    beta = bundle["beta"]
    accum = int(cfg["grad_accum_steps"])
    epochs = int(cfg.get("epochs", 1))
    max_grad = float(cfg.get("max_grad_norm", 1.0))
    device = next(model.parameters()).device
    params = trainable_parameters(model)

    results_dir = repo_path(cfg["results_dir"]) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    run_config = {
        "run_name": run_name, "beta": beta, "dataset": bundle["dataset_path"], "n_examples": len(bundle["rows"]),
        "n_filtered_long_prompts": bundle["n_filtered_long_prompts"], "max_examples": max_examples,
        "epochs": epochs, "batch_size": int(cfg["batch_size"]), "grad_accum_steps": accum,
        "effective_batch": int(cfg["batch_size"]) * accum, "learning_rate": float(cfg["learning_rate"]),
        "lr_schedule": "constant", "weight_decay": float(cfg.get("weight_decay", 0.0)), "max_grad_norm": max_grad,
        "max_sequence_length": int(cfg["max_sequence_length"]), "lora": cfg["lora"], "seed": int(cfg["seed"]),
        "base_model": cfg["base_model"], "dtype": cfg.get("dtype"), "output": str(output),
    }
    save_json(results_dir / "run_config.json", run_config)
    print(f"[dpo:{run_name}] {len(bundle['rows'])} pairs, beta={beta}, effective batch {run_config['effective_batch']}")

    reset_peak_memory()
    t0 = time.perf_counter()
    n_batches = len(loader)
    step = 0
    seen = 0
    window = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "cr": 0.0, "rr": 0.0, "n": 0}
    opt.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        for i, (chosen, rejected) in enumerate(loader):
            window_start = (i // accum) * accum
            window_size = min(accum, n_batches - window_start)

            with torch.no_grad(), reference_mode(model):
                ref_c, ref_r = pair_logps(model, chosen, rejected, tok.pad_token_id, device)
            pol_c, pol_r = pair_logps(model, chosen, rejected, tok.pad_token_id, device)
            loss, info = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
            (loss / window_size).backward()

            bs = chosen["input_ids"].shape[0]
            seen += bs
            window["loss"] += float(loss.item()) * bs
            window["acc"] += float(info["preference_accuracy"]) * bs
            window["margin"] += float(info["dpo_margin_mean"]) * bs
            window["cr"] += float(info["chosen_reward_mean"]) * bs
            window["rr"] += float(info["rejected_reward_mean"]) * bs
            window["n"] += bs

            if (i + 1) % accum == 0 or (i + 1) == n_batches:
                gn = grad_norm(params)
                torch.nn.utils.clip_grad_norm_(params, max_grad)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                n = max(window["n"], 1)
                rec = {
                    "step": step, "epoch": epoch, "examples_seen": seen,
                    "loss": window["loss"] / n, "preference_accuracy": window["acc"] / n,
                    "dpo_margin": window["margin"] / n, "chosen_reward": window["cr"] / n,
                    "rejected_reward": window["rr"] / n, "grad_norm": gn,
                    "lr": opt.param_groups[0]["lr"], "elapsed_s": time.perf_counter() - t0,
                }
                append_jsonl(log_path, rec)
                if step % int(cfg.get("log_every", 1)) == 0:
                    print(f"[dpo:{run_name}] step {step}/{math.ceil(n_batches / accum) * epochs} loss {rec['loss']:.4f} "
                          f"acc {rec['preference_accuracy']:.3f} margin {rec['dpo_margin']:.3f} gn {gn:.3f}")
                window = {k: 0.0 for k in window}
                window["n"] = 0

    model.save_pretrained(str(output))
    tok.save_pretrained(str(output))
    summary = {**run_config, "optimizer_steps": step, "examples_seen": seen,
               "wall_clock_s": time.perf_counter() - t0, "peak_vram_gib": peak_memory_gib()}
    save_json(results_dir / "train_summary.json", summary)
    print(f"[dpo:{run_name}] saved adapter to {output}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
