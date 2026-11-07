import argparse
import json
import numpy as np
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch


FLOW_METHOD = "Flow-Abstain"

OOD_DATASET_FILES = {
    "squad2": ("SQuAD2", "squad2_unanswerable.jsonl", True),
    "truthfulqa": ("TruthfulQA", "truthfulqa_validation.jsonl", True),
    "popqa": ("PopQA", "popqa.jsonl", False),
    "ambigqa": ("AmbigQA", "ambigqa.jsonl", False),
}


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item.strip()]


def variant_dir_name(component, setting):
    clean = str(setting).lower().replace(" ", "_").replace("(", "").replace(")", "")
    clean = clean.replace("$", "").replace("\\", "").replace("/", "_")
    return f"{component}_{clean}"


def run_command(command, dry_run):
    print(" ".join(str(part) for part in command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def build_run_command(args, output, pca_dim, maf_blocks, layer):
    command = [
        sys.executable,
        "run_experiments.py",
        "--model",
        args.model,
        "--model-name",
        args.model_name,
        "--layer",
        str(layer),
        "--pca-dim",
        str(pca_dim),
        "--maf-blocks",
        str(maf_blocks),
        "--maf-epochs",
        str(args.maf_epochs),
        "--max-id",
        str(args.max_id),
        "--max-ood",
        str(args.max_ood),
        "--train-frac",
        str(args.train_frac),
        "--split-seed",
        str(args.split_seed),
        "--datasets",
        *args.datasets,
        "--batch-size",
        str(args.batch_size),
        "--data",
        args.data,
        "--output",
        str(output),
        "--device",
        args.device,
    ]
    if args.skip_output_baselines:
        command.append("--skip-output-baselines")
    if args.use_substring_match:
        command.append("--use-substring-match")
    if args.use_token_f1_match:
        command.extend(["--use-token-f1-match", "--token-f1-threshold", str(args.token_f1_threshold)])
    if args.system_prompt is not None:
        command.extend(["--system-prompt", args.system_prompt])
    if args.no_chat_template:
        command.append("--no-chat-template")
    return command


def average_metrics(metrics_path, benchmarks):
    metrics = pd.read_csv(metrics_path)
    flow = metrics[metrics["method"] == FLOW_METHOD].copy()
    if benchmarks:
        flow = flow[flow["benchmark"].isin(benchmarks)]
    if flow.empty:
        raise ValueError(f"No {FLOW_METHOD} rows found in {metrics_path}.")
    return float(flow["auroc"].mean()), float(flow["fpr95"].mean()), sorted(flow["benchmark"].unique())


def prepare_flow_only_context(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from run_experiments import (
        answer_match_reason,
        build_prompt,
        generate_answers,
        read_jsonl,
        split_indices,
    )

    data = Path(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        device_map="auto" if args.device.startswith("cuda") else None,
        trust_remote_code=True,
    )
    if not args.device.startswith("cuda"):
        model.to(args.device)
    model.eval()

    prompt_kwargs = {
        "tokenizer": tokenizer,
        "system_prompt": args.system_prompt,
        "use_chat_template": not args.no_chat_template,
    }
    trivia = read_jsonl(data / "triviaqa_validation.jsonl", args.max_id)
    trivia_prompts = [build_prompt(row["question"], **prompt_kwargs) for row in trivia]
    answers = generate_answers(model, tokenizer, trivia_prompts, args.batch_size, args.device)
    match_reasons = [
        answer_match_reason(
            pred,
            row["answers"],
            use_substring_match=args.use_substring_match,
            use_token_f1_match=args.use_token_f1_match,
            token_f1_threshold=args.token_f1_threshold,
        )
        for pred, row in zip(answers, trivia)
    ]
    keep = [i for i, reason in enumerate(match_reasons) if reason != "none"]
    keep_set = set(keep)
    miss = [i for i in range(len(trivia)) if i not in keep_set]

    counts = {reason: match_reasons.count(reason) for reason in sorted(set(match_reasons))}
    match_summary = {
        "use_substring_match": args.use_substring_match,
        "use_token_f1_match": args.use_token_f1_match,
        "token_f1_threshold": args.token_f1_threshold,
        "counts": counts,
        "matched": len(keep),
        "unmatched": len(miss),
        "train_frac": args.train_frac,
        "split_seed": args.split_seed,
    }
    if len(keep) < 100:
        raise RuntimeError(
            f"Only {len(keep)} matched TriviaQA examples found; increase --max-id or check match settings."
        )
    if len(miss) < 100:
        raise RuntimeError(f"Only {len(miss)} incorrect TriviaQA examples found; increase --max-id or check prompt/model.")

    id_train_idx, id_eval_idx = split_indices(keep, args.train_frac, args.split_seed)
    miss_train_idx, miss_eval_idx = split_indices(miss, args.train_frac, args.split_seed + 1)
    if len(id_eval_idx) < 100:
        raise RuntimeError(
            f"Only {len(id_eval_idx)} held-out matched TriviaQA examples; increase --max-id or lower --train-frac."
        )
    if len(miss_eval_idx) < 100:
        raise RuntimeError(
            f"Only {len(miss_eval_idx)} held-out incorrect TriviaQA examples; increase --max-id or lower --train-frac."
        )

    datasets = {}
    for dataset in args.datasets:
        benchmark_name, filename, required = OOD_DATASET_FILES[dataset]
        path = data / filename
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Requested dataset {dataset!r} not found at {path}")
            print(f"Skipping requested optional dataset {dataset!r}: {path} does not exist")
            continue

        rows = read_jsonl(path, args.max_ood)
        prompts = [build_prompt(row["question"], row.get("context"), **prompt_kwargs) for row in rows]
        datasets[benchmark_name] = prompts

    return {
        "model": model,
        "tokenizer": tokenizer,
        "model_name": args.model_name or Path(args.model).name,
        "match_summary": match_summary,
        "id_train_prompts": [trivia_prompts[i] for i in id_train_idx],
        "id_eval_prompts": [trivia_prompts[i] for i in id_eval_idx],
        "miss_eval_prompts": [trivia_prompts[i] for i in miss_eval_idx],
        "datasets": datasets,
        "hidden_cache": {},
    }


def cached_hidden(context, args, benchmark, prompts, layer):
    from run_experiments import extract_hidden

    key = (benchmark, layer)
    if key not in context["hidden_cache"]:
        context["hidden_cache"][key] = extract_hidden(
            context["model"],
            context["tokenizer"],
            prompts,
            layer,
            args.batch_size,
            args.device,
        )
    return context["hidden_cache"][key]


def run_flow_only_experiment(args, output, pca_dim, maf_blocks, layer, context):
    from flow_abstain import FlowAbstain, FlowAbstainConfig
    from run_experiments import metric_row

    output.mkdir(parents=True, exist_ok=True)
    with (output / "triviaqa_match_summary.json").open("w", encoding="utf-8") as f:
        json.dump(context["match_summary"], f, indent=2)

    id_train_hidden = cached_hidden(context, args, "TriviaQA Train", context["id_train_prompts"], layer)
    id_eval_hidden = cached_hidden(context, args, "TriviaQA", context["id_eval_prompts"], layer)
    miss_eval_hidden = cached_hidden(context, args, "TriviaQA Incorrect", context["miss_eval_prompts"], layer)

    cfg = FlowAbstainConfig(
        pca_dim=min(pca_dim, id_train_hidden.shape[0] - 1, id_train_hidden.shape[1]),
        maf_blocks=maf_blocks,
        epochs=args.maf_epochs,
        device=args.device,
    )
    flow = FlowAbstain(cfg).fit(id_train_hidden)
    flow.save(output / "flow_abstain.pt")

    metrics = []
    score_rows = []
    id_scores = flow.score(id_eval_hidden)
    miss_scores = flow.score(miss_eval_hidden)
    trivia_y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(miss_scores))])
    metrics.append(
        metric_row(
            context["model_name"],
            "TriviaQA",
            FLOW_METHOD,
            trivia_y_true,
            np.concatenate([id_scores, miss_scores]),
        )
    )

    for benchmark, scores in [("TriviaQA", id_scores), ("TriviaQA Incorrect", miss_scores)]:
        for index, score in enumerate(scores):
            score_rows.append(
                {
                    "model": context["model_name"],
                    "benchmark": benchmark,
                    "method": FLOW_METHOD,
                    "index": index,
                    "score": float(score),
                }
            )

    for benchmark_name, prompts in context["datasets"].items():
        hidden = cached_hidden(context, args, benchmark_name, prompts, layer)
        ood_scores = flow.score(hidden)
        y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
        scores = np.concatenate([id_scores, ood_scores])
        metrics.append(metric_row(context["model_name"], benchmark_name, FLOW_METHOD, y_true, scores))
        for index, score in enumerate(ood_scores):
            score_rows.append(
                {
                    "model": context["model_name"],
                    "benchmark": benchmark_name,
                    "method": FLOW_METHOD,
                    "index": index,
                    "score": float(score),
                }
            )

    metrics_df = pd.DataFrame(metrics)
    scores_df = pd.DataFrame(score_rows)
    metrics_df.to_csv(output / "metrics.csv", index=False)
    scores_df.to_csv(output / "scores.csv", index=False)
    print(metrics_df.sort_values(["benchmark", "method"]).to_string(index=False))


def write_latex_table(rows, output, caption, label):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        "",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        "",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"\textbf{Component} & \textbf{Setting} & \textbf{Avg. AUROC} & \textbf{Avg. FPR95} \\",
        r"\midrule",
    ]

    component_order = ["PCA (dim $d$)", "MAF (blocks $K$)", r"Layer ($\ell$)"]
    for component_index, component in enumerate(component_order):
        group = [row for row in rows if row["component"] == component]
        for index, row in enumerate(group):
            component_cell = rf"\multirow{{{len(group)}}}{{*}}{{{component}}}" if index == 0 else ""
            setting = row["setting"]
            auroc = f"{row['avg_auroc']:.1f}"
            fpr95 = f"{row['avg_fpr95']:.1f}"
            if row["is_baseline"]:
                setting = rf"\textbf{{{setting}}}"
                auroc = rf"\textbf{{{auroc}}}"
                fpr95 = rf"\textbf{{{fpr95}}}"
            lines.append(rf"{component_cell} & {setting} & {auroc} & {fpr95} \\")
        if component_index != len(component_order) - 1:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    Path(output).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run PCA/MAF/layer ablations for Flow-Abstain and "
            "summarize average Flow-Abstain AUROC into a LaTeX table."
        )
    )
    parser.add_argument("--model", required=True, help="Model path or HuggingFace id.")
    parser.add_argument("--model-name", default="Qwen3-8B-Instruct")
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--output-root", default="results/qwen3_8b_ablation")
    parser.add_argument("--table-output", default=None)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-id", type=int, default=12000)
    parser.add_argument("--max-ood", type=int, default=3000)
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Fraction of TriviaQA correct examples used to train Flow-Abstain.",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(OOD_DATASET_FILES),
        choices=sorted(OOD_DATASET_FILES),
        help="OOD benchmarks to evaluate. TriviaQA remains the fixed ID source.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--maf-epochs", type=int, default=200)
    parser.add_argument("--baseline-pca-dim", type=int, default=128)
    parser.add_argument("--baseline-maf-blocks", type=int, default=8)
    parser.add_argument("--baseline-layer", type=int, default=18)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--pca-dims", default="none,64,128,256")
    parser.add_argument("--maf-blocks", default="2,4,8,16")
    parser.add_argument("--layers", default="8,16,24,32")
    parser.add_argument(
        "--system-prompt",
        default="You are a QA system. Answer with only the final short answer. Do not explain.",
        help="System prompt passed to run_experiments.py.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Pass --no-chat-template to run_experiments.py.",
    )
    parser.add_argument(
        "--use-substring-match",
        action="store_true",
        help="Pass --use-substring-match to run_experiments.py.",
    )
    parser.add_argument(
        "--use-token-f1-match",
        action="store_true",
        help="Pass --use-token-f1-match to run_experiments.py.",
    )
    parser.add_argument("--token-f1-threshold", type=float, default=0.8)
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Benchmarks to average. Defaults to every Flow-Abstain row in metrics.csv.",
    )
    parser.add_argument(
        "--skip-output-baselines",
        action="store_true",
        default=True,
        help="Skip Perplexity/P(True)/Semantic Entropy during ablation runs.",
    )
    parser.add_argument(
        "--include-output-baselines",
        action="store_false",
        dest="skip_output_baselines",
        help="Run output-space baselines too. This is much slower and not needed for the ablation table.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun variants even if metrics.csv exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--legacy-run-experiments",
        action="store_true",
        help="Use the old behavior that calls run_experiments.py and computes all baselines.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.token_f1_threshold <= 1.0:
        raise ValueError("--token-f1-threshold must be between 0 and 1.")
    if not 0.0 < args.train_frac < 1.0:
        raise ValueError("--train-frac must be between 0 and 1.")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    table_output = Path(args.table_output) if args.table_output else output_root / "ablation_table.tex"
    csv_output = Path(args.csv_output) if args.csv_output else output_root / "ablation_summary.csv"

    pca_settings = []
    for item in args.pca_dims.split(","):
        item = item.strip().lower()
        if item == "none":
            pca_settings.append(("None (4096-D)", args.hidden_dim))
        else:
            pca_settings.append((item, int(item)))
    maf_settings = parse_int_list(args.maf_blocks)
    layer_settings = parse_int_list(args.layers)

    variants = []
    for setting_label, pca_dim in pca_settings:
        variants.append(
            {
                "component": "PCA (dim $d$)",
                "setting": setting_label,
                "pca_dim": pca_dim,
                "maf_blocks": args.baseline_maf_blocks,
                "layer": args.baseline_layer,
                "is_baseline": pca_dim == args.baseline_pca_dim,
            }
        )
    for maf_blocks in maf_settings:
        variants.append(
            {
                "component": "MAF (blocks $K$)",
                "setting": str(maf_blocks),
                "pca_dim": args.baseline_pca_dim,
                "maf_blocks": maf_blocks,
                "layer": args.baseline_layer,
                "is_baseline": maf_blocks == args.baseline_maf_blocks,
            }
        )
    for layer in layer_settings:
        if layer == args.baseline_layer:
            setting = f"{layer} (mid.)"
        elif layer < args.baseline_layer:
            setting = f"{layer} (early)"
        elif layer >= max(layer_settings):
            setting = f"{layer} (final)"
        else:
            setting = f"{layer} (late)"
        variants.append(
            {
                "component": r"Layer ($\ell$)",
                "setting": setting,
                "pca_dim": args.baseline_pca_dim,
                "maf_blocks": args.baseline_maf_blocks,
                "layer": layer,
                "is_baseline": layer == args.baseline_layer,
            }
        )

    flow_context = None
    run_cache = {}
    rows = []
    for variant in variants:
        key = (variant["pca_dim"], variant["maf_blocks"], variant["layer"])
        if key not in run_cache:
            run_name = variant_dir_name("run", f"pca{key[0]}_maf{key[1]}_layer{key[2]}")
            if args.use_substring_match or args.use_token_f1_match:
                match_parts = []
                if args.use_substring_match:
                    match_parts.append("substring")
                if args.use_token_f1_match:
                    threshold = f"{args.token_f1_threshold:g}".replace(".", "p")
                    match_parts.append(f"f1_{threshold}")
                run_name = f"{run_name}_{'_'.join(match_parts)}"
            output = output_root / run_name
            metrics_path = output / "metrics.csv"
            if args.force or not metrics_path.exists():
                output.mkdir(parents=True, exist_ok=True)
                if args.legacy_run_experiments:
                    command = build_run_command(
                        args,
                        output,
                        pca_dim=variant["pca_dim"],
                        maf_blocks=variant["maf_blocks"],
                        layer=variant["layer"],
                    )
                    run_command(command, args.dry_run)
                else:
                    print(
                        "Flow-only ablation "
                        f"--output {output} --pca-dim {variant['pca_dim']} "
                        f"--maf-blocks {variant['maf_blocks']} --layer {variant['layer']}",
                        flush=True,
                    )
                    if not args.dry_run:
                        if flow_context is None:
                            flow_context = prepare_flow_only_context(args)
                        run_flow_only_experiment(
                            args,
                            output,
                            pca_dim=variant["pca_dim"],
                            maf_blocks=variant["maf_blocks"],
                            layer=variant["layer"],
                            context=flow_context,
                        )
            run_cache[key] = metrics_path

        metrics_path = run_cache[key]
        if args.dry_run:
            avg_auroc = float("nan")
            avg_fpr95 = float("nan")
            used_benchmarks = []
        else:
            avg_auroc, avg_fpr95, used_benchmarks = average_metrics(metrics_path, args.benchmarks)
        rows.append(
            {
                "component": variant["component"],
                "setting": variant["setting"],
                "pca_dim": variant["pca_dim"],
                "maf_blocks": variant["maf_blocks"],
                "layer": variant["layer"],
                "avg_auroc": avg_auroc,
                "avg_fpr95": avg_fpr95,
                "is_baseline": variant["is_baseline"],
                "metrics": str(metrics_path),
                "benchmarks": ",".join(used_benchmarks),
            }
        )

    summary = pd.DataFrame(rows)
    caption = (
        r"\textbf{Ablation study on PCA dimensionality, MAF depth, and extraction layer} "
        r"using Qwen-3-8B-Instruct. We report average AUROC and FPR95 (\%) across QA benchmarks."
    )

    print(summary[["component", "setting", "avg_auroc", "avg_fpr95", "metrics"]].to_string(index=False))
    if args.dry_run:
        print("\nDry run only; no summary files were written.")
    else:
        summary.to_csv(csv_output, index=False)
        write_latex_table(rows, table_output, caption, "tab:ablation")
        print(f"\nWrote {csv_output}")
        print(f"Wrote {table_output}")
    print(
        "\nNote: PCA setting 'None (4096-D)' is implemented as --pca-dim 4096 because "
        "run_experiments.py does not currently expose a true no-PCA mode."
    )


if __name__ == "__main__":
    main()
