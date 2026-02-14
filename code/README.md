# Cloned Repositories

## CAA
- URL: https://github.com/nrimsky/CAA
- Purpose: Official implementation of Contrastive Activation Addition (CAA) for steering Llama 2.
- Location: `code/CAA/`
- Key files: `generate_vectors.py`, `prompting_with_steering.py`, `normalize_vectors.py`, `vectors/`
- Notes: Requires HuggingFace Llama 2 access and API keys; includes datasets used for steering vectors.

## Selective Steering
- URL: https://github.com/knoveleng/steering
- Purpose: Official implementation of Selective Steering (norm-preserving, layer-selective steering).
- Location: `code/steering/`
- Key files: `steering/pipeline.py`, `configs/`, `artifacts/`
- Notes: GPU required; supports Qwen, Llama, Gemma; uses calibrated steering planes.

## MLSAE
- URL: https://github.com/tim-lawson/mlsae
- Purpose: Multi-layer SAE training and analysis for residual-stream features.
- Location: `code/mlsae/`
- Key files: `mlsae/model/`, `train.py`, `mlsae/analysis/`, `figures/`
- Notes: Provides pretrained MLSAEs and analysis scripts for residual stream distributions.

## TransformerLens
- URL: https://github.com/TransformerLensOrg/TransformerLens
- Purpose: Mechanistic interpretability library for extracting and editing activations.
- Location: `code/transformerlens/`
- Key files: `transformer_lens/`, `demos/`
- Notes: Useful for residual stream access and activation patching/steering experiments.

## Endogenous Steering Resistance
- URL: https://github.com/agencyenterprise/endogenous-steering-resistance
- Purpose: Code for ESR experiments using SAE steering and self-correction analysis.
- Location: `code/endogenous-steering-resistance/`
- Key files: `experiment_01_esr.py`, `experiment_02_multi_boost.py`, `plotting/`
- Notes: Requires large GPU resources and vllm-interp; Llama-3.3-70B used in paper.

## HC3 Plus Data + Detector
- URL: https://github.com/suu990901/chatgpt-comparison-detection-HC3-Plus
- Purpose: HC3 Plus dataset and detection baselines (RoBERTa, Tk-instruct).
- Location: `code/hc3-plus/`
- Key files: `data/`, `train_english_roberta.sh`, `train_chinese_roberta.sh`
- Notes: Data copied to `datasets/HC3-Plus/` for experiment use.
