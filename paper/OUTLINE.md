# Outline: Sounds-Like-AI Residual Direction Paper

## Title
- Working: "A Linear Residual-Stream Direction for 'Sounds Like AI' Style in LLMs"

## Abstract
- Problem: AI-sounding style is a user-facing issue; unclear if it is linearly encoded.
- Gap: Prior steering shows linear directions for behaviors, but not for AI-sounding style tied to HC3.
- Approach: Probe residual activations, compute mean-difference direction, steer generation at discriminative layer.
- Results: 97.8% probe accuracy at layer 12; steering shifts AI-likeness (+0.048 / -0.156) with small n.
- Significance: Evidence for linear style encoding; motivates larger, multi-model validation.

## Introduction
- Hook: AI-sounding outputs reduce perceived quality; controlling style matters.
- Importance: Steering methods can adjust behavior without retraining.
- Gap: AI-sounding style not directly tested; layer dependence underexplored.
- Approach: Use HC3, probe layers, compute mean-diff direction, steer at discriminative layer; refer to method figure.
- Quantitative preview: 97.8% accuracy; AI-plus/AI-minus shifts; p-values non-significant in pilot.
- Contributions (3-4 bullets)

## Related Work (by theme)
- Linear steering (CAA, Selective Steering) -> baseline approach.
- Representation steering methods (RePS, SRS, SVF) -> alternatives for robustness.
- Identifiability and resistance (non-identifiability, ESR) -> caution for interpretation.
- Datasets (HC3, HC3-Plus) -> operationalizing AI-sounding.

## Methodology
- Problem formulation: binary labels y in {0,1} for AI vs human.
- Activation extraction: mean-pooled residual states at layers {0,12,23}.
- Probe: logistic regression; metrics accuracy/precision/recall/F1.
- Direction: v = mean(h_AI) - mean(h_H); steer by h' = h + alpha v.
- Layer selection: pick best by test accuracy.
- Generation + evaluation: 10 prompts; probe scores; GPT-4.1 judge.
- Baselines: no steering only (random direction not run).

## Results
- Table: probe accuracy per layer (bold best).
- Table: AI-likeness shifts (base vs AI-plus/AI-minus).
- Figure: distributions of probe scores (score_distributions.png).
- Statistical tests: t-tests with p-values provided.

## Discussion
- Interpret results: linear separability + directional control.
- Limitations: small pilot, single model, single layer, limited judge size.
- Implications: evidence for style direction; need robustness.

## Conclusion
- Summary of findings and future work.

## Figures/Tables Plan
- Fig 1: Method overview schematic.
- Fig 2: Score distributions (existing plot).
- Table 1: Probe results.
- Table 2: Steering results.

## Citations Needed
- CAA, Selective Steering, RePS, SVF, identifiability, ESR, SRS, MLSAE, HC3, HC3-Plus.
