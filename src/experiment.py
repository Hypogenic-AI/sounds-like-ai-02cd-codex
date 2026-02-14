import os
import json
import random
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_path: str = "datasets/HC3"
    output_dir: str = "results"
    max_samples_per_class: int = 300
    max_answer_len: int = 256
    batch_size: int = 16
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    layers_to_probe: Tuple[int, int, int] = (0, 10, 23)  # will be updated after model load
    generation_prompts: int = 10
    gen_max_new_tokens: int = 64
    gen_temperature: float = 0.7
    steering_alpha: float = 2.0


# -----------------------------
# Data utilities
# -----------------------------

def flatten_hc3(ds) -> Tuple[List[str], List[int], List[str]]:
    texts = []
    labels = []
    sources = []
    for row in ds:
        question = row["question"]
        source = row.get("source", "unknown")
        # human answers
        for ans in row["human_answers"]:
            texts.append(ans)
            labels.append(0)
            sources.append(source)
        # chatgpt answers
        for ans in row["chatgpt_answers"]:
            texts.append(ans)
            labels.append(1)
            sources.append(source)
    return texts, labels, sources


def sample_balanced(texts, labels, sources, max_per_class, seed=42):
    rng = np.random.default_rng(seed)
    idx_h = [i for i, y in enumerate(labels) if y == 0]
    idx_a = [i for i, y in enumerate(labels) if y == 1]
    rng.shuffle(idx_h)
    rng.shuffle(idx_a)
    idx_h = idx_h[:max_per_class]
    idx_a = idx_a[:max_per_class]
    idx = idx_h + idx_a
    rng.shuffle(idx)
    texts_s = [texts[i] for i in idx]
    labels_s = [labels[i] for i in idx]
    sources_s = [sources[i] for i in idx]
    return texts_s, labels_s, sources_s


# -----------------------------
# Embedding extraction
# -----------------------------

def mean_pool_last_hidden(last_hidden, attention_mask):
    # last_hidden: [B, T, H], attention_mask: [B, T]
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def extract_hidden_states(model, tokenizer, texts, layers, max_len, batch_size, device):
    model.eval()
    all_layer_states = {layer: [] for layer in layers}

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = outputs.hidden_states  # tuple length = n_layers + 1 (embeddings)

        for layer in layers:
            hs = hidden_states[layer + 1]  # layer index -> hidden_states offset by embedding
            pooled = mean_pool_last_hidden(hs, attention_mask)
            all_layer_states[layer].append(pooled.cpu().float())

    for layer in layers:
        all_layer_states[layer] = torch.cat(all_layer_states[layer], dim=0).numpy()

    return all_layer_states


# -----------------------------
# Steering utilities
# -----------------------------

def get_layer_modules(model):
    # Try common architectures: Llama/Qwen2-like
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model architecture for hooking layers.")


def register_addition_hook(layer_module, direction, alpha):
    def hook_fn(module, input, output):
        dir_vec = direction.to(dtype=output[0].dtype if isinstance(output, tuple) else output.dtype)
        if isinstance(output, tuple):
            hs = output[0]
            hs = hs + alpha * dir_vec
            return (hs,) + output[1:]
        return output + alpha * dir_vec

    return layer_module.register_forward_hook(hook_fn)


def generate_with_hook(model, tokenizer, prompt, layer_idx, direction, alpha, max_new_tokens, temperature):
    device = next(model.parameters()).device
    layer_modules = get_layer_modules(model)
    # direction shape [H]
    direction = direction.to(device=device, dtype=next(model.parameters()).dtype)

    hook = register_addition_hook(layer_modules[layer_idx], direction, alpha)
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    finally:
        hook.remove()

    return text


# -----------------------------
# Main experiment
# -----------------------------

def main():
    cfg = Config()
    set_seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "plots"), exist_ok=True)

    # Load dataset
    ds = load_from_disk(cfg.dataset_path)["train"]
    texts, labels, sources = flatten_hc3(ds)

    # Basic data quality stats
    lengths = [len(t.split()) for t in texts]
    data_stats = {
        "total_samples": len(texts),
        "mean_length_words": float(np.mean(lengths)),
        "median_length_words": float(np.median(lengths)),
        "missing_texts": int(np.sum([t is None or t == "" for t in texts])),
        "class_counts": {"human": int(sum(1 for y in labels if y == 0)), "ai": int(sum(1 for y in labels if y == 1))},
    }

    # Balanced subsample
    texts_s, labels_s, sources_s = sample_balanced(texts, labels, sources, cfg.max_samples_per_class, cfg.seed)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts_s, labels_s, test_size=0.3, random_state=cfg.seed, stratify=labels_s
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=cfg.seed, stratify=y_temp
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model for hidden states (avoid logits to reduce memory)
    model = AutoModel.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=None,
        trust_remote_code=True,
    )
    model.to(cfg.device)

    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("Could not read num_hidden_layers from model config.")
    cfg.layers_to_probe = (0, num_layers // 2, num_layers - 1)

    # Extract embeddings
    layer_states_train = extract_hidden_states(model, tokenizer, X_train, cfg.layers_to_probe, cfg.max_answer_len, cfg.batch_size, cfg.device)
    layer_states_val = extract_hidden_states(model, tokenizer, X_val, cfg.layers_to_probe, cfg.max_answer_len, cfg.batch_size, cfg.device)
    layer_states_test = extract_hidden_states(model, tokenizer, X_test, cfg.layers_to_probe, cfg.max_answer_len, cfg.batch_size, cfg.device)

    results = {
        "config": cfg.__dict__,
        "data_stats": data_stats,
        "probe_results": {},
    }

    # Train/eval probes and compute directions
    for layer in cfg.layers_to_probe:
        Xtr = layer_states_train[layer]
        Xva = layer_states_val[layer]
        Xte = layer_states_test[layer]

        clf = LogisticRegression(max_iter=200, n_jobs=-1)
        clf.fit(Xtr, y_train)
        preds = clf.predict(Xte)
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary")

        # Mean difference direction (AI - human)
        mean_ai = Xtr[np.array(y_train) == 1].mean(axis=0)
        mean_h = Xtr[np.array(y_train) == 0].mean(axis=0)
        direction = mean_ai - mean_h

        results["probe_results"][str(layer)] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "direction_norm": float(np.linalg.norm(direction)),
        }

        # Save direction
        np.save(os.path.join(cfg.output_dir, f"direction_layer_{layer}.npy"), direction)

    # Choose best layer
    best_layer = max(results["probe_results"].items(), key=lambda x: x[1]["accuracy"])[0]
    best_layer = int(best_layer)

    # Free base model before generation
    del model
    torch.cuda.empty_cache()

    # Load causal LM for steering generation
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=None,
        trust_remote_code=True,
    )
    model.to(cfg.device)

    # Steering experiment
    # Use a subset of questions as prompts
    prompts = []
    for row in ds.select(range(cfg.generation_prompts)):
        prompts.append(f"Question: {row['question']}\nAnswer:")

    direction = torch.tensor(np.load(os.path.join(cfg.output_dir, f"direction_layer_{best_layer}.npy")), dtype=torch.float32)
    direction = direction / (direction.norm() + 1e-8)

    generations = []
    for prompt in prompts:
        base = generate_with_hook(
            model,
            tokenizer,
            prompt,
            best_layer,
            direction,
            alpha=0.0,
            max_new_tokens=cfg.gen_max_new_tokens,
            temperature=cfg.gen_temperature,
        )
        steered = generate_with_hook(
            model,
            tokenizer,
            prompt,
            best_layer,
            direction,
            alpha=cfg.steering_alpha,
            max_new_tokens=cfg.gen_max_new_tokens,
            temperature=cfg.gen_temperature,
        )
        unsteered = generate_with_hook(
            model,
            tokenizer,
            prompt,
            best_layer,
            direction,
            alpha=-cfg.steering_alpha,
            max_new_tokens=cfg.gen_max_new_tokens,
            temperature=cfg.gen_temperature,
        )
        generations.append({
            "prompt": prompt,
            "base": base,
            "ai_plus": steered,
            "ai_minus": unsteered,
        })

    # Score generations with probe at best layer
    def score_texts(texts):
        emb = extract_hidden_states(model, tokenizer, texts, [best_layer], cfg.max_answer_len, cfg.batch_size, cfg.device)[best_layer]
        clf = LogisticRegression(max_iter=200, n_jobs=-1)
        clf.fit(layer_states_train[best_layer], y_train)
        scores = clf.predict_proba(emb)[:, 1]
        return scores

    base_texts = [g["base"] for g in generations]
    plus_texts = [g["ai_plus"] for g in generations]
    minus_texts = [g["ai_minus"] for g in generations]

    base_scores = score_texts(base_texts)
    plus_scores = score_texts(plus_texts)
    minus_scores = score_texts(minus_texts)

    results["steering"] = {
        "best_layer": best_layer,
        "score_means": {
            "base": float(np.mean(base_scores)),
            "ai_plus": float(np.mean(plus_scores)),
            "ai_minus": float(np.mean(minus_scores)),
        },
        "score_stds": {
            "base": float(np.std(base_scores)),
            "ai_plus": float(np.std(plus_scores)),
            "ai_minus": float(np.std(minus_scores)),
        },
    }

    # Save generations and scores
    with open(os.path.join(cfg.output_dir, "model_outputs", "generations.json"), "w") as f:
        json.dump(generations, f, indent=2)

    np.save(os.path.join(cfg.output_dir, "evals", "base_scores.npy"), base_scores)
    np.save(os.path.join(cfg.output_dir, "evals", "plus_scores.npy"), plus_scores)
    np.save(os.path.join(cfg.output_dir, "evals", "minus_scores.npy"), minus_scores)

    # Plot score distributions
    plt.figure(figsize=(7, 4))
    sns.kdeplot(base_scores, label="base")
    sns.kdeplot(plus_scores, label="ai_plus")
    sns.kdeplot(minus_scores, label="ai_minus")
    plt.title("AI-likeness score distributions")
    plt.xlabel("AI-likeness (probe probability)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "plots", "score_distributions.png"), dpi=200)
    plt.close()

    # Save results
    results["timestamp"] = datetime.now().isoformat()
    with open(os.path.join(cfg.output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Environment info
    env_info = {
        "python": os.popen("python -V").read().strip(),
        "torch": torch.__version__,
        "device": cfg.device,
    }
    with open(os.path.join(cfg.output_dir, "env.json"), "w") as f:
        json.dump(env_info, f, indent=2)


if __name__ == "__main__":
    main()
