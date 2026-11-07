import argparse
import json
from pathlib import Path

from datasets import load_dataset


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_available(candidates):
    last_error = None
    for args, kwargs in candidates:
        try:
            return load_dataset(*args, **kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(str(last_error))


def normalize_triviaqa(example):
    aliases = []
    answer = example.get("answer", {})
    if isinstance(answer, dict):
        aliases.extend(answer.get("aliases") or [])
        if answer.get("value"):
            aliases.append(answer["value"])
        if answer.get("normalized_value"):
            aliases.append(answer["normalized_value"])
    return {
        "id": example.get("question_id") or example.get("id"),
        "question": example["question"],
        "answers": sorted(set(a for a in aliases if a)),
    }


def normalize_squad(example):
    return {
        "id": example.get("id"),
        "question": example["question"],
        "context": example.get("context", ""),
        "answers": example.get("answers", {}).get("text", []),
        "is_unanswerable": len(example.get("answers", {}).get("text", [])) == 0,
    }


def normalize_question_only(example):
    question = (
        example.get("question")
        or example.get("query")
        or example.get("input")
        or example.get("ambiguous_question")
    )
    if question is None:
        return None
    return {"id": example.get("id") or example.get("question_id"), "question": question}


def parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_popqa(example, min_subject_popularity):
    question = example.get("question")
    subject_popularity = parse_int(example.get("subject_popularity") or example.get("s_pop"))
    if question is None or subject_popularity is None:
        return None
    if subject_popularity <= min_subject_popularity:
        return None

    return {
        "id": example.get("id"),
        "question": question,
        "answers": example.get("answers") or example.get("possible_answers"),
        "subject": example.get("subject") or example.get("subj"),
        "object": example.get("object") or example.get("obj"),
        "property": example.get("property") or example.get("prop"),
        "subject_popularity": subject_popularity,
        "object_popularity": parse_int(example.get("object_popularity") or example.get("o_pop")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw")
    parser.add_argument(
        "--popqa-min-subject-popularity",
        type=int,
        default=1000,
        help="Keep PopQA examples with subject_popularity rank greater than this value.",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    trivia = first_available([
        (("trivia_qa", "rc.nocontext"), {"split": "validation"}),
        (("mandarjoshi/trivia_qa", "rc.nocontext"), {"split": "validation"}),
    ])
    write_jsonl((normalize_triviaqa(x) for x in trivia), out / "triviaqa_validation.jsonl")

    squad = load_dataset("squad_v2", split="validation")
    unanswerable = (normalize_squad(x) for x in squad if len(x.get("answers", {}).get("text", [])) == 0)
    write_jsonl(unanswerable, out / "squad2_unanswerable.jsonl")

    truthful = load_dataset("truthful_qa", "generation", split="validation")
    write_jsonl((normalize_question_only(x) for x in truthful), out / "truthfulqa_validation.jsonl")

    try:
        popqa = first_available([
            (("csv",), {
                "data_files": "https://huggingface.co/datasets/akariasai/PopQA/resolve/main/test.tsv",
                "delimiter": "\t",
                "split": "train",
            }),
        ])
        rows = (
            row
            for row in (
                normalize_popqa(x, min_subject_popularity=args.popqa_min_subject_popularity)
                for x in popqa
            )
            if row is not None
        )
        write_jsonl(rows, out / "popqa.jsonl")
    except Exception as exc:
        print(f"Skipping popqa.jsonl: {exc}")

    optional = {
        "ambigqa.jsonl": [
            (("ambig_qa", "light"), {"split": "validation"}),
            (("sewon/ambig_qa", "light"), {"split": "validation"}),
        ],
    }
    for filename, candidates in optional.items():
        try:
            ds = first_available(candidates)
            rows = (row for row in (normalize_question_only(x) for x in ds) if row is not None)
            write_jsonl(rows, out / filename)
        except Exception as exc:
            print(f"Skipping {filename}: {exc}")

    print(f"Wrote datasets to {out}")


if __name__ == "__main__":
    main()
