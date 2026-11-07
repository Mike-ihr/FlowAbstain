import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from flow_abstain import FlowAbstain, FlowAbstainConfig


DEFAULT_SYSTEM_PROMPT = "You are a QA system. Answer with only the final short answer. Do not explain."

OOD_DATASET_FILES = {
    "squad2": ("SQuAD2", "squad2_unanswerable.jsonl", True),
    "truthfulqa": ("TruthfulQA", "truthfulqa_validation.jsonl", True),
    "popqa": ("PopQA", "popqa.jsonl", False),
    "ambigqa": ("AmbigQA", "ambigqa.jsonl", False),
}


def read_jsonl(path, limit=None):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def split_indices(indices, train_frac, seed):
    indices = np.asarray(indices)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    split = int(round(train_frac * len(shuffled)))
    split = min(max(1, split), len(shuffled) - 1)
    return shuffled[:split].tolist(), shuffled[split:].tolist()


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


def exact_match(prediction, answers):
    pred = normalize_text(prediction)
    return any(pred == normalize_text(a) for a in answers if a)


def strip_thinking(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def first_answer_line(text):
    text = strip_thinking(text)
    for line in text.split("\n"):
        line = line.strip()
        if line:
            return line
    return ""


def apply_chat_template(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def last_nonpad_indices(attention_mask):
    return attention_mask.shape[1] - 1 - attention_mask.flip(dims=[1]).argmax(dim=1)


def token_f1(prediction, answer):
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(answer).split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def answer_match_reason(
    prediction,
    answers,
    use_substring_match=False,
    use_token_f1_match=False,
    token_f1_threshold=0.8,
):
    pred = normalize_text(prediction)
    best_f1 = 0.0
    for answer in answers:
        gold = normalize_text(answer)
        if not gold:
            continue

        if pred == gold:
            return "exact"

        if use_substring_match and len(gold) >= 2 and gold in pred:
            return "substring"

        if use_token_f1_match:
            best_f1 = max(best_f1, token_f1(pred, gold))

    if use_token_f1_match and best_f1 >= token_f1_threshold:
        return "token_f1"
    return "none"


def build_prompt(question, context=None, tokenizer=None, system_prompt=DEFAULT_SYSTEM_PROMPT, use_chat_template=True):
    if context:
        user_content = (
            "Answer the question using only the provided context.\n"
            f"Context: {context}\n"
            f"Question: {question}"
        )
    else:
        user_content = f"Question: {question}"

    if use_chat_template and tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return apply_chat_template(tokenizer, messages)

    if context:
        return (
            f"System: {system_prompt}\n"
            "Answer the question using only the provided context.\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            "Answer:"
        )
    return f"System: {system_prompt}\nQuestion: {question}\nAnswer:"


@torch.no_grad()
def extract_hidden(model, tokenizer, prompts, layer, batch_size, device):
    states = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="hidden states"):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[layer]
        last_idx = last_nonpad_indices(enc["attention_mask"])
        rows = hidden[torch.arange(hidden.shape[0], device=device), last_idx]
        states.append(rows.float().cpu().numpy())
    return np.concatenate(states, axis=0)


@torch.no_grad()
def generate_answers(model, tokenizer, prompts, batch_size, device, max_new_tokens=128):
    answers = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="answers"):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        answers.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return [first_answer_line(a) for a in answers]


@torch.no_grad()
def generated_answer_nll(model, tokenizer, prompts, answers, batch_size, device):
    scores = []
    pairs = list(zip(prompts, answers))
    for i in tqdm(range(0, len(pairs), batch_size), desc="perplexity"):
        batch = pairs[i : i + batch_size]
        full_text = [p + " " + a for p, a in batch]
        prompt_enc = tokenizer(
            [p for p, _ in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        enc = tokenizer(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=544,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, use_cache=False)
        logits = out.logits[:, :-1, :]
        labels = enc["input_ids"][:, 1:]
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape(labels.shape)
        for row, (prompt, answer) in enumerate(batch):
            prompt_len = int(prompt_enc["attention_mask"][row].sum().item())
            full_len = int(enc["attention_mask"][row].sum().item())
            pad_len = enc["attention_mask"].shape[1] - full_len
            answer_start = max(0, pad_len + prompt_len - 1)
            answer_mask = enc["attention_mask"][row, 1:].bool()
            answer_mask[:answer_start] = False
            loss = token_loss[row][answer_mask].mean()
            scores.append(-float(loss.cpu()))
    return np.asarray(scores, dtype=np.float32)


@torch.no_grad()
def p_true_scores(model, tokenizer, prompts, answers, batch_size, device):
    scores = []
    true_ids = tokenizer(" True", add_special_tokens=False)["input_ids"]
    false_ids = tokenizer(" False", add_special_tokens=False)["input_ids"]
    true_id = true_ids[-1]
    false_id = false_ids[-1]
    eval_prompts = [
        (
            f"{prompt} {answer}\n\n"
            "Is the answer above correct? Reply with True or False.\n"
            "Verdict:"
        )
        for prompt, answer in zip(prompts, answers)
    ]
    for i in tqdm(range(0, len(eval_prompts), batch_size), desc="P(True)"):
        batch = eval_prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=768)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, use_cache=False)
        last_idx = last_nonpad_indices(enc["attention_mask"])
        logits = out.logits[torch.arange(out.logits.shape[0], device=device), last_idx]
        two = torch.stack([logits[:, true_id], logits[:, false_id]], dim=-1)
        scores.extend(torch.softmax(two, dim=-1)[:, 0].float().cpu().numpy())
    return np.asarray(scores, dtype=np.float32)


def substring_cluster_counts(keys, min_substring_len=2):
    if not keys:
        return []

    parent = list(range(len(keys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            left = keys[i]
            right = keys[j]
            shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
            if left == right or (len(shorter) >= min_substring_len and shorter in longer):
                union(i, j)

    counts = Counter(find(i) for i in range(len(keys)))
    return list(counts.values())


@torch.no_grad()
def lexical_semantic_entropy(model, tokenizer, prompts, batch_size, device, samples=10, max_new_tokens=128):
    entropies = []
    for prompt in tqdm(prompts, desc="semantic entropy"):
        enc = tokenizer([prompt] * samples, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        keys = [normalize_text(first_answer_line(text)) for text in texts]
        cluster_counts = substring_cluster_counts(keys)
        probs = np.asarray(cluster_counts, dtype=np.float32) / float(samples)
        entropies.append(-float(np.sum(probs * np.log(probs + 1e-12))))
    return -np.asarray(entropies, dtype=np.float32)


def fpr_at_tpr(y_true, scores, target_tpr=0.95):
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = np.where(tpr >= target_tpr)[0]
    return float(fpr[valid[0]]) if len(valid) else 1.0


def metric_row(model_name, benchmark, method, y_true, scores):
    return {
        "model": model_name,
        "benchmark": benchmark,
        "method": method,
        "auroc": 100.0 * roc_auc_score(y_true, scores),
        "auprc": 100.0 * average_precision_score(y_true, scores),
        "fpr95": 100.0 * fpr_at_tpr(y_true, scores),
    }


def plot_likelihood(scores_df, output):
    flow_df = scores_df[scores_df["method"] == "Flow-Abstain"].copy()
    flow_df = flow_df[np.isfinite(flow_df["score"])]
    if flow_df.empty:
        return
    lo, hi = np.percentile(flow_df["score"], [10, 90])
    bins = np.linspace(lo, hi, 60)
    colors = {
        "TriviaQA": "tab:green",
        "TriviaQA Incorrect": "tab:gray",
        "SQuAD2": "tab:blue",
        "PopQA": "tab:orange",
        "TruthfulQA": "tab:red",
        "AmbigQA": "tab:purple",
    }

    plt.figure(figsize=(7, 4.5))
    for name, group in flow_df.groupby("benchmark"):
        if name == "TriviaQA":
            continue
        values = group["score"].clip(lo, hi)
        color = colors.get(name)
        counts, edges, _ = plt.hist(
            values,
            bins=bins,
            density=True,
            alpha=0.35,
            label=name,
            color=color,
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.plot(centers, counts, linewidth=1.8, color=color)
    id_scores = flow_df[flow_df["benchmark"] == "TriviaQA"]["score"].clip(lo, hi)
    color = colors["TriviaQA"]
    counts, edges, _ = plt.hist(
        id_scores,
        bins=bins,
        density=True,
        alpha=0.45,
        label="TriviaQA ID",
        color=color,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.plot(centers, counts, linewidth=2.0, color=color)
    plt.xlim(lo, hi)
    plt.xlabel("Log-likelihood")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "likelihood_histogram.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-id", type=int, default=12000)
    parser.add_argument("--max-ood", type=int, default=3000)
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Fraction of TriviaQA correct/incorrect examples used to train representation baselines.",
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
    parser.add_argument("--maf-blocks", type=int, default=8)
    parser.add_argument("--maf-epochs", type=int, default=200)
    parser.add_argument("--semantic-samples", type=int, default=10)
    parser.add_argument(
        "--use-substring-match",
        action="store_true",
        help="Treat a prediction as correct if a normalized gold answer appears inside it.",
    )
    parser.add_argument(
        "--use-token-f1-match",
        action="store_true",
        help="Treat a prediction as correct if token F1 against any gold answer exceeds the threshold.",
    )
    parser.add_argument("--token-f1-threshold", type=float, default=0.8)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Use a plain text prompt instead of tokenizer.apply_chat_template.",
    )
    parser.add_argument("--skip-output-baselines", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0.0 <= args.token_f1_threshold <= 1.0:
        raise ValueError("--token-f1-threshold must be between 0 and 1.")
    if not 0.0 < args.train_frac < 1.0:
        raise ValueError("--train-frac must be between 0 and 1.")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name or args.model.split("/")[-1]

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

    data = Path(args.data)
    trivia = read_jsonl(data / "triviaqa_validation.jsonl", args.max_id)
    prompt_kwargs = {
        "tokenizer": tokenizer,
        "system_prompt": args.system_prompt,
        "use_chat_template": not args.no_chat_template,
    }
    trivia_prompts = [build_prompt(x["question"], **prompt_kwargs) for x in trivia]
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
    match_counts = Counter(match_reasons)
    print(
        "TriviaQA match counts: "
        f"exact={match_counts['exact']}, "
        f"substring={match_counts['substring']}, "
        f"token_f1={match_counts['token_f1']}, "
        f"none={match_counts['none']}"
    )
    match_summary = {
        "use_substring_match": args.use_substring_match,
        "use_token_f1_match": args.use_token_f1_match,
        "token_f1_threshold": args.token_f1_threshold,
        "counts": dict(match_counts),
        "matched": len(keep),
        "unmatched": len(miss),
        "train_frac": args.train_frac,
        "split_seed": args.split_seed,
    }
    with (output / "triviaqa_match_summary.json").open("w", encoding="utf-8") as f:
        json.dump(match_summary, f, indent=2)
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

    id_train_prompts = [trivia_prompts[i] for i in id_train_idx]
    id_eval_rows = [trivia[i] for i in id_eval_idx]
    id_eval_prompts = [trivia_prompts[i] for i in id_eval_idx]
    miss_train_prompts = [trivia_prompts[i] for i in miss_train_idx]
    miss_eval_prompts = [trivia_prompts[i] for i in miss_eval_idx]
    id_train_hidden = extract_hidden(model, tokenizer, id_train_prompts, args.layer, args.batch_size, args.device)
    id_eval_hidden = extract_hidden(model, tokenizer, id_eval_prompts, args.layer, args.batch_size, args.device)
    miss_train_hidden = extract_hidden(model, tokenizer, miss_train_prompts, args.layer, args.batch_size, args.device)
    miss_eval_hidden = extract_hidden(model, tokenizer, miss_eval_prompts, args.layer, args.batch_size, args.device)

    cfg = FlowAbstainConfig(
        pca_dim=min(args.pca_dim, id_train_hidden.shape[0] - 1, id_train_hidden.shape[1]),
        maf_blocks=args.maf_blocks,
        epochs=args.maf_epochs,
        device=args.device,
    )
    flow = FlowAbstain(cfg).fit(id_train_hidden)
    flow.save(output / "flow_abstain.pt")

    pca = PCA(n_components=cfg.pca_dim, whiten=True, random_state=0).fit(id_train_hidden)
    id_train_z = pca.transform(id_train_hidden)
    id_eval_z = pca.transform(id_eval_hidden)
    cov = EmpiricalCovariance().fit(id_train_z)
    gmm = GaussianMixture(n_components=min(16, len(id_train_z)), covariance_type="full", random_state=0).fit(id_train_z)
    probe_train_x = np.concatenate([id_train_hidden, miss_train_hidden], axis=0)
    probe_train_y = np.concatenate([np.ones(len(id_train_hidden)), np.zeros(len(miss_train_hidden))])
    probe_pca = PCA(n_components=cfg.pca_dim, whiten=True, random_state=0).fit(probe_train_x)
    probe = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        probe_pca.transform(probe_train_x), probe_train_y
    )
    id_probe_scores = probe.predict_proba(probe_pca.transform(id_eval_hidden))[:, 1]
    miss_z = pca.transform(miss_eval_hidden)
    miss_flow_scores = flow.score(miss_eval_hidden)
    miss_maha_scores = -cov.mahalanobis(miss_z)
    miss_gmm_scores = gmm.score_samples(miss_z)
    miss_probe_scores = probe.predict_proba(probe_pca.transform(miss_eval_hidden))[:, 1]

    id_answers = [answers[i] for i in id_eval_idx]
    miss_answers = [answers[i] for i in miss_eval_idx]
    benchmarks = {"TriviaQA": (id_eval_rows, id_eval_prompts, id_eval_hidden, id_answers)}
    for dataset in args.datasets:
        benchmark_name, filename, required = OOD_DATASET_FILES[dataset]
        path = data / filename
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Requested dataset {dataset!r} not found at {path}")
            print(f"Skipping requested optional dataset {dataset!r}: {path} does not exist")
            continue
        benchmarks[benchmark_name] = (read_jsonl(path, args.max_ood), None, None, None)

    metrics = []
    score_rows = []
    cached_id_output_scores = None
    trivia_y_true = np.concatenate([np.ones(len(id_eval_hidden)), np.zeros(len(miss_eval_hidden))])
    trivia_method_scores = {
        "Flow-Abstain": np.concatenate([flow.score(id_eval_hidden), miss_flow_scores]),
        "Mahalanobis": np.concatenate([-cov.mahalanobis(id_eval_z), miss_maha_scores]),
        "GMM-16": np.concatenate([gmm.score_samples(id_eval_z), miss_gmm_scores]),
        "Linear Probe": np.concatenate([id_probe_scores, miss_probe_scores]),
    }
    if not args.skip_output_baselines:
        id_output_scores = {
            "Perplexity": generated_answer_nll(
                model, tokenizer, id_eval_prompts, id_answers, args.batch_size, args.device
            ),
            "P(True)": p_true_scores(
                model, tokenizer, id_eval_prompts, id_answers, args.batch_size, args.device
            ),
            "Semantic Entropy": lexical_semantic_entropy(
                model,
                tokenizer,
                id_eval_prompts,
                args.batch_size,
                args.device,
                samples=args.semantic_samples,
            ),
        }
        miss_output_scores = {
            "Perplexity": generated_answer_nll(
                model, tokenizer, miss_eval_prompts, miss_answers, args.batch_size, args.device
            ),
            "P(True)": p_true_scores(
                model, tokenizer, miss_eval_prompts, miss_answers, args.batch_size, args.device
            ),
            "Semantic Entropy": lexical_semantic_entropy(
                model,
                tokenizer,
                miss_eval_prompts,
                args.batch_size,
                args.device,
                samples=args.semantic_samples,
            ),
        }
        for method, scores in id_output_scores.items():
            trivia_method_scores[method] = np.concatenate([scores, miss_output_scores[method]])
        cached_id_output_scores = id_output_scores
    for method, scores in trivia_method_scores.items():
        metrics.append(metric_row(model_name, "TriviaQA", method, trivia_y_true, scores))
    for method, values in [
        ("Flow-Abstain", miss_flow_scores),
        ("Mahalanobis", miss_maha_scores),
        ("GMM-16", miss_gmm_scores),
    ]:
        for i, score in enumerate(values):
            score_rows.append(
                {
                    "model": model_name,
                    "benchmark": "TriviaQA Incorrect",
                    "method": method,
                    "index": i,
                    "score": float(score),
                }
            )

    for benchmark, (rows, prompts, hidden, bench_answers) in benchmarks.items():
        if prompts is None:
            prompts = [build_prompt(x["question"], x.get("context"), **prompt_kwargs) for x in rows]
        if hidden is None:
            hidden = extract_hidden(model, tokenizer, prompts, args.layer, args.batch_size, args.device)
        if bench_answers is None and not args.skip_output_baselines:
            bench_answers = generate_answers(model, tokenizer, prompts, args.batch_size, args.device)

        flow_scores = flow.score(hidden)
        z = pca.transform(hidden)
        maha_scores = -cov.mahalanobis(z)
        gmm_scores = gmm.score_samples(z)

        for method, values in [
            ("Flow-Abstain", flow_scores),
            ("Mahalanobis", maha_scores),
            ("GMM-16", gmm_scores),
        ]:
            for i, score in enumerate(values):
                score_rows.append(
                    {
                        "model": model_name,
                        "benchmark": benchmark,
                        "method": method,
                        "index": i,
                        "score": float(score),
                    }
                )

        if benchmark == "TriviaQA":
            continue

        y_true = np.concatenate([np.ones(len(id_eval_hidden)), np.zeros(len(hidden))])
        probe_scores = np.concatenate([
            id_probe_scores,
            probe.predict_proba(probe_pca.transform(hidden))[:, 1],
        ])
        method_scores = {
            "Flow-Abstain": np.concatenate([flow.score(id_eval_hidden), flow_scores]),
            "Mahalanobis": np.concatenate([-cov.mahalanobis(id_eval_z), maha_scores]),
            "GMM-16": np.concatenate([gmm.score_samples(id_eval_z), gmm_scores]),
            "Linear Probe": probe_scores,
        }
        if not args.skip_output_baselines:
            if cached_id_output_scores is None:
                cached_id_output_scores = {
                    "Perplexity": generated_answer_nll(
                        model, tokenizer, id_eval_prompts, id_answers, args.batch_size, args.device
                    ),
                    "P(True)": p_true_scores(
                        model, tokenizer, id_eval_prompts, id_answers, args.batch_size, args.device
                    ),
                    "Semantic Entropy": lexical_semantic_entropy(
                        model,
                        tokenizer,
                        id_eval_prompts,
                        args.batch_size,
                        args.device,
                        samples=args.semantic_samples,
                    ),
                }
            output_scores = {
                "Perplexity": generated_answer_nll(
                    model, tokenizer, prompts, bench_answers, args.batch_size, args.device
                ),
                "P(True)": p_true_scores(
                    model, tokenizer, prompts, bench_answers, args.batch_size, args.device
                ),
                "Semantic Entropy": lexical_semantic_entropy(
                    model,
                    tokenizer,
                    prompts,
                    args.batch_size,
                    args.device,
                    samples=args.semantic_samples,
                ),
            }
            for method, ood_scores in output_scores.items():
                method_scores[method] = np.concatenate([cached_id_output_scores[method], ood_scores])
        for method, scores in method_scores.items():
            metrics.append(metric_row(model_name, benchmark, method, y_true, scores))

    metrics_df = pd.DataFrame(metrics)
    scores_df = pd.DataFrame(score_rows)
    metrics_df.to_csv(output / "metrics.csv", index=False)
    scores_df.to_csv(output / "scores.csv", index=False)
    plot_likelihood(scores_df, output)
    print(metrics_df.sort_values(["benchmark", "method"]).to_string(index=False))


if __name__ == "__main__":
    main()
