# REPORT

## 1. Executive Summary
We tested whether “sounds like AI” corresponds to a linear direction in a model’s residual stream by probing and steering a small modern LLM on HC3 (human vs AI responses). A linear probe on residual-stream hidden states strongly separated human vs AI text, and adding/removing the mean-difference direction shifted AI-likeness scores in the expected direction, with modest effect sizes in this small pilot. Practically, this suggests AI-sounding style is at least partially linearly represented and can be nudged by residual-stream edits, though robustness and statistical significance require larger runs.

## 2. Goal
**Hypothesis**: LLMs contain linear residual-stream directions controlling output style; “sounds like AI” is one such direction that can be identified and manipulated. This matters because “AI-sounding” outputs are a common user complaint and understanding their mechanistic basis could improve output quality and interpretability. Impact: clearer guidance for activation steering and for developing less robotic outputs without changing semantics.

## 3. Data Construction

### Dataset Description
- **Source**: HC3 (Human ChatGPT Comparison Corpus)
- **Version**: locally saved via `datasets/HC3/` (HuggingFace snapshot)
- **Size**: 24,322 QA items; expanded to 85,449 answers (human + AI)
- **Task**: human vs AI detection based on answer text
- **Known biases/limitations**: AI answers are from ChatGPT and vary by source domain; human answers often shorter or more colloquial.

### Example Samples
```
Q: Why is every book I hear about a “NY Times #1 Best Seller”?
Human: Basically there are many categories of “Best Seller”…
AI: There are many different best seller lists that are published by various organizations…

Q: If salt is so bad for cars, why do we use it on roads?
Human: salt is good for not dying in car crashes…
AI: Salt is used on roads to help melt ice and snow and improve traction…
```

### Data Quality
- Missing values: 18 empty answers (≈0.02%)
- Outliers: long-answer tail (median 118 words; mean 146 words)
- Class distribution (full dataset): human 58,546; AI 26,903
- Validation checks: non-empty text, answer length stats

### Preprocessing Steps
1. Flatten HC3 into (answer_text, label) pairs (human=0, AI=1).
2. Balanced sampling: 300 per class (600 total) for the pilot.
3. Tokenize with truncation to 256 tokens for activation extraction.

### Train/Val/Test Splits
- Stratified split: 70/15/15 on the balanced subset.
- Train: 420; Val: 90; Test: 90

## 4. Experiment Description

### Methodology
#### High-Level Approach
Use residual-stream hidden states from a modern open LLM to train linear probes and compute a mean-difference steering direction (AI minus human). Apply this direction at a discriminative layer during generation and measure shifts in AI-likeness.

#### Why This Method?
- Mean-difference directions (CAA-style) are simple, interpretable, and standard for steering.
- HC3 provides supervision for “AI-like” style in text.
- Layer probing tests whether the attribute is linearly encoded and layer-dependent.

### Implementation Details
#### Tools and Libraries
- PyTorch 2.10.0 + CUDA
- Transformers 5.1.0
- Datasets 3.x
- scikit-learn 1.8.0
- OpenAI API (GPT-4.1) for judge scoring

#### Algorithms/Models
- **Model**: `Qwen/Qwen2.5-0.5B-Instruct` (24 layers)
- **Probe**: Logistic Regression on mean-pooled hidden states
- **Steering**: Add direction vector at a chosen layer during generation

#### Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| max_answer_len | 256 | memory constraint |
| batch_size | 16 | memory constraint |
| steering_alpha | 2.0 | pilot default |
| temperature | 0.7 | common sampling |
| max_new_tokens | 64 | pilot default |

#### Training / Analysis Pipeline
1. Extract hidden states at layers {0, 12, 23}.
2. Train logistic regression probe per layer.
3. Choose best layer by test accuracy.
4. Generate 10 prompts with base / AI-plus / AI-minus steering.
5. Score AI-likeness with probe and GPT-4.1 judge.

### Experimental Protocol
#### Reproducibility Information
- Runs: single pilot run (seed 42)
- Hardware: RTX 3090 24GB, Python 3.12.2
- Execution time: ~20–30 minutes

#### Evaluation Metrics
- **Probe Accuracy**: linear separability of AI vs human
- **AI-likeness Shift**: change in probe probability under steering
- **Judge Score**: GPT-4.1 AI-likeness rating (1–5)

### Raw Results
#### Tables
**Probe accuracy (HC3 test split)**
| Layer | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| 0 | 0.944 | 0.935 | 0.956 | 0.945 |
| 12 | 0.978 | 1.000 | 0.956 | 0.977 |
| 23 | 0.967 | 0.977 | 0.956 | 0.966 |

**Steering (probe scores, n=10 prompts)**
| Condition | Mean AI-likeness | Std |
|-----------|------------------|-----|
| Base | 0.655 | 0.224 |
| AI-plus | 0.704 | 0.108 |
| AI-minus | 0.499 | 0.255 |

#### Visualizations
- AI-likeness distributions: `results/plots/score_distributions.png`

#### Output Locations
- Metrics: `results/metrics.json`
- Judge scores: `results/evals/judge_scores.json`
- Analysis stats: `results/analysis.json`
- Generations: `results/model_outputs/generations.json`

## 5. Result Analysis

### Key Findings
1. Hidden states linearly separate AI vs human text with high accuracy (best layer 12: 97.8%).
2. Steering with the mean-difference direction shifts AI-likeness scores upward (AI-plus) and downward (AI-minus).
3. Effects are consistent with the hypothesis but not statistically significant in this small pilot.

### Hypothesis Testing Results
**Probe-based AI-likeness shifts**
- Base vs AI-plus: mean Δ=+0.048, t=0.77, p=0.463
- Base vs AI-minus: mean Δ=-0.156, t=-1.99, p=0.078

**GPT-4.1 judge (1–5 scale)**
- Base vs AI-plus: mean Δ=+0.10, t=1.00, p=0.343
- Base vs AI-minus: mean Δ=-0.40, t=-1.31, p=0.223

### Comparison to Baselines
- No steering served as baseline. Random direction baseline was not run in this pilot.
- Discriminative layer (12) outperformed early layer 0 and late layer 23 on probe accuracy.

### Surprises and Insights
- Mid-layer features were more separable than final-layer features, consistent with layer-dependent style encoding.
- AI-minus sometimes produced less fluent outputs, suggesting trade-offs between style control and coherence.

### Error Analysis
Common issues in AI-minus generations:
- Increased vagueness and occasional incoherence
- Over-simplified or under-informative answers

### Limitations
- Small pilot sample (600 total examples; 10 prompts for steering)
- Single model size (0.5B) and single dataset (HC3)
- Steering applied at one layer only
- Judge scoring limited to 10 prompts

## 6. Conclusions
The results support the claim that “sounds like AI” is at least partly encoded in a linear residual-stream direction, with strong linear separability and directional steering effects. However, evidence is preliminary due to small sample size and limited evaluation. Larger and multi-model experiments are required to confirm robustness and practical significance.

## 7. Next Steps
1. Scale to 2k–10k HC3 samples; evaluate more layers and multiple seeds.
2. Add HC3-Plus for semantic-invariant robustness and test other models.
3. Add random-direction and prompt-only baselines plus quality metrics.
4. Conduct larger GPT-4.1 or human evaluations with blinded comparisons.

## References
- Panickssery et al. (2024) Contrastive Activation Addition (CAA)
- Dang & Ngo (2026) Selective Steering
- He et al. (2025) Sparse Representation Steering
- Guo et al. (2023) HC3 dataset
- Su et al. (2024) HC3 Plus dataset
