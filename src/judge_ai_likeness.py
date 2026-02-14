import os
import json
import time
from typing import Dict
from openai import OpenAI

INPUT_PATH = "results/model_outputs/generations.json"
OUTPUT_PATH = "results/evals/judge_scores.json"
MODEL = "gpt-4.1"

client = OpenAI()

SYSTEM = "You are an expert writing-style evaluator."
PROMPT = (
    "Rate how much the answer sounds like it was written by an AI assistant, "
    "ignoring factual correctness. Use a 1-5 integer scale: 1=very human, 5=very AI. "
    "Return JSON with keys: score (int), rationale (short)."
)


def judge(text: str) -> Dict:
    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{PROMPT}\n\nAnswer:\n{text}"},
        ],
        temperature=0.0,
        max_output_tokens=120,
    )
    content = resp.output_text
    # Basic parse fallback
    try:
        data = json.loads(content)
        return data
    except Exception:
        return {"score": None, "rationale": content}


def main():
    with open(INPUT_PATH, "r") as f:
        gens = json.load(f)

    results = []
    for row in gens:
        entry = {"prompt": row["prompt"], "scores": {}}
        for key in ["base", "ai_plus", "ai_minus"]:
            text = row[key]
            data = judge(text)
            entry["scores"][key] = data
            time.sleep(0.2)
        results.append(entry)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
