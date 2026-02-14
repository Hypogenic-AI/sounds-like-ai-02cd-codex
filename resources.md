# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project, including papers, datasets, and code repositories.

## Papers
Total papers downloaded: 10

| Title | Authors | Year | File | Key Info |
|---|---|---|---|---|
| Steering Llama 2 via Contrastive Activation Addition | Panickssery et al. | 2024 | `papers/2312.06681_contrastive_activation_addition.pdf` | Residual-stream steering vectors (CAA) |
| Selective Steering: Norm-Preserving Control Through Discriminative Layer Selection | Dang & Ngo | 2026 | `papers/2601.19375_selective_steering.pdf` | Norm-preserving, layer-selective steering |
| Improved Representation Steering for Language Models | Wu et al. | 2025 | `papers/2505.20809_representation_projection_steering.pdf` | RePS objective, AXBENCH evaluation |
| Steering Vector Fields for Context-Aware Inference-Time Control in LLMs | Li et al. | 2026 | `papers/2602.01654_steering_vector_fields.pdf` | Context-dependent steering |
| On the Identifiability of Steering Vectors in LLMs | Venkatesh & Kurapath | 2026 | `papers/2602.06801_identifiability_of_steering_vectors.pdf` | Identifiability limits |
| Endogenous Resistance to Activation Steering in LLMs | McKenzie et al. | 2026 | `papers/2602.06941_endogenous_resistance_activation_steering.pdf` | ESR self-correction under steering |
| Interpretable LLM Guardrails via Sparse Representation Steering | He et al. | 2025 | `papers/2503.16851_sparse_representation_steering.pdf` | SAE-based sparse steering |
| Residual Stream Analysis with Multi-Layer SAEs | Lawson et al. | 2025 | `papers/2409.04185_residual_stream_analysis_multi_layer_saes.pdf` | MLSAE latent dynamics across layers |
| HC3: Human ChatGPT Comparison Corpus | Guo et al. | 2023 | `papers/2301.07597_hc3_human_chatgpt_comparison_corpus.pdf` | Human vs AI QA dataset |
| HC3 Plus: Semantic-Invariant Human ChatGPT Comparison Corpus | Su et al. | 2024 | `papers/2309.02731_hc3_plus.pdf` | Adds summarization/translation/paraphrase |

See `papers/README.md` for detailed descriptions.

## Datasets
Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|---|---|---|---|---|---|
| HC3 | HuggingFace `Hello-SimpleAI/HC3` | 24,322 QA items (~42 MB) | Human vs AI detection | `datasets/HC3/` | Saved with `save_to_disk` |
| HC3 Plus | GitHub `suu990901/chatgpt-comparison-detection-HC3-Plus` | ~73 MB | Human vs AI detection (QA + semantic-invariant) | `datasets/HC3-Plus/` | JSONL files for en/zh |

See `datasets/README.md` for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 6

| Name | URL | Purpose | Location | Notes |
|---|---|---|---|---|
| CAA | https://github.com/nrimsky/CAA | CAA steering vectors | `code/CAA/` | Llama 2 steering experiments |
| Selective Steering | https://github.com/knoveleng/steering | Norm-preserving steering | `code/steering/` | Calibrated planes, GPU required |
| MLSAE | https://github.com/tim-lawson/mlsae | Multi-layer SAEs | `code/mlsae/` | Residual stream analysis |
| TransformerLens | https://github.com/TransformerLensOrg/TransformerLens | Mech interp library | `code/transformerlens/` | Activation caching and editing |
| Endogenous Steering Resistance | https://github.com/agencyenterprise/endogenous-steering-resistance | ESR experiments | `code/endogenous-steering-resistance/` | vllm-interp dependency |
| HC3 Plus | https://github.com/suu990901/chatgpt-comparison-detection-HC3-Plus | Dataset + detector baselines | `code/hc3-plus/` | Data copied to datasets |

See `code/README.md` for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Focused on residual-stream steering, activation editing, SAE-based steering, and AI-vs-human detection datasets.
- Prioritized recent (2024–2026) arXiv preprints for steering methods and datasets.

### Selection Criteria
- Direct relevance to residual-stream directions, steering reliability, or AI-vs-human style detection.
- Availability of code and datasets for reproducibility.
- Mix of method papers (CAA, Selective Steering, SVF, SRS) and data papers (HC3, HC3 Plus).

### Challenges Encountered
- The paper-finder service was unresponsive; manual arXiv search and direct PDF download were used.
- HuggingFace HC3 required `trust_remote_code=True` and an explicit config (`all`).

### Gaps and Workarounds
- No single standard “sounds like AI” benchmark exists; HC3/HC3 Plus provide the closest proxy.
- Steering identifiability limits suggest using multiple validation environments or sparsity constraints.

## Recommendations for Experiment Design

1. **Primary dataset(s)**: HC3 for initial direction discovery; HC3 Plus for robustness on semantic-invariant tasks.
2. **Baseline methods**: CAA and Selective Steering; add SRS for SAE-based comparison.
3. **Evaluation metrics**: Human-vs-AI detection accuracy, perplexity/fluency metrics, and judge-based quality ratings.
4. **Code to adapt/reuse**: TransformerLens for activation access; CAA/Selective Steering repos for steering pipelines; MLSAE for feature discovery across layers.

## Research Execution Log (2026-02-14)

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset: HC3 (balanced subset of 300 human + 300 AI answers)
- Probing layers: 0, 12, 23 (best: 12 with 97.8% accuracy)
- Steering: mean-difference direction at layer 12; evaluated on 10 prompts
- Outputs: `results/metrics.json`, `results/analysis.json`, `results/plots/score_distributions.png`
- Judge evaluation: GPT-4.1 scores stored in `results/evals/judge_scores.json`
