# Planning

## Motivation & Novelty Assessment

### Why This Research Matters
“Sounds like AI” is a common, user-relevant complaint about LLM outputs; if it corresponds to a controllable linear direction, we can reduce that style without degrading content. Understanding whether this style is encoded as a residual-stream direction informs both interpretability and practical steering techniques. This has implications for human-LLM interaction quality and for interpretability claims about linear feature structure.

### Gap in Existing Work
Prior steering work shows many behaviors are linearly steerable in residual streams (CAA, Selective Steering, SRS), but none directly test “sounds like AI” as a style attribute tied to human-vs-AI detection datasets. The literature also highlights identifiability limits and layer dynamics, suggesting that a naive single direction may be unstable or layer-dependent for style.

### Our Novel Contribution
We explicitly operationalize “sounds like AI” using HC3/HC3-Plus human-vs-AI data, derive residual-stream directions for this attribute, and test whether adding/removing those directions shifts AI-likeness in generated text. We also examine layer dependence and discriminative-layer selection to assess stability of the direction.

### Experiment Justification
- Experiment 1: Learn and evaluate a residual-stream “AI-likeness” direction from HC3. Needed to test if the attribute is linearly separable in activations.
- Experiment 2: Apply the direction during generation and measure changes in AI-likeness vs content quality. Needed to test causal controllability.
- Experiment 3: Layer-selection and ablation (single vs multi-layer; remove vs add) to test stability and identifiability limits.

## Research Question
Is “sounds like AI” represented as a linear direction in the residual stream that can be identified and causally manipulated to change output style?

## Background and Motivation
LLM steering via residual-stream directions can modulate behavior with minimal performance loss. However, style attributes like “AI-sounding” are underexplored, and may be context- or layer-dependent. Using HC3/HC3-Plus allows a concrete operationalization of “AI-like” style for both probing and evaluation.

## Hypothesis Decomposition
- H1: Residual-stream activations contain a linearly separable signal for AI vs human responses.
- H2: Adding the AI-likeness direction increases AI-detection scores; removing it decreases scores.
- H3: The direction is more stable/effective in certain layers (discriminative layers) than uniformly across all layers.

## Proposed Methodology

### Approach
Use a small, modern open LLM with activation access (TransformerLens) to extract residual activations for HC3 samples. Compute mean-difference directions (CAA-style) between AI and human responses. Evaluate linear separability and apply the direction during generation. Validate AI-likeness shifts with a trained classifier and a GPT-4.1 judge on a small subset.

### Experimental Steps
1. Load HC3 dataset and create train/val/test splits with balanced AI/human labels (rationale: controlled evaluation).
2. Extract residual-stream activations at multiple layers for each sample (rationale: test linear structure, layer dependence).
3. Train linear probes and compute mean-difference directions; evaluate accuracy on held-out set (rationale: quantify linear separability).
4. Identify discriminative layers by projection separation and probe performance (rationale: layer selection stability).
5. Generate responses to prompts with/without steering; apply direction addition or removal at selected layers (rationale: causal test).
6. Evaluate AI-likeness via classifier scores and GPT-4.1 judgments on a subset; assess fluency/quality with perplexity proxy or repetition metrics (rationale: style change vs content degradation).

### Baselines
- No steering (prompt-only generation)
- Random direction addition (control)
- Uniform steering across all layers vs discriminative-layer steering
- Direction removal (projection ablation)

### Evaluation Metrics
- AI-vs-human classification accuracy (probe) on held-out HC3
- Mean AI-likeness score shift under steering (classifier logit/probability)
- GPT-4.1 judge AI-likeness rating on a subset
- Fluency proxies: repetition rate, length-normalized perplexity (if feasible)

### Statistical Analysis Plan
- Two-sided paired t-test or Wilcoxon signed-rank on AI-likeness score shifts (steered vs baseline)
- Effect sizes (Cohen’s d for paired differences)
- Multiple-comparison correction if comparing multiple layers/conditions (Benjamini–Hochberg)
- Confidence intervals via bootstrap for mean shifts

## Expected Outcomes
Support for hypothesis if: (a) linear probes significantly separate AI vs human, (b) steering reliably shifts AI-likeness scores in expected direction without large fluency degradation, and (c) specific layers show stronger, more stable control.

## Timeline and Milestones
- Phase 0–1: Planning and resource review (completed here)
- Phase 2: Environment setup + data checks (30–45 min)
- Phase 3: Implementation of extraction/probing/steering (1–2 hrs)
- Phase 4: Experiments and runs (1–2 hrs)
- Phase 5: Analysis and visualization (45–60 min)
- Phase 6: Documentation and validation (45–60 min)

## Potential Challenges
- Access to a suitable open model with TransformerLens compatibility
- Compute time for activation extraction
- Identifiability instability across layers
- API cost/latency for GPT-4.1 judge

## Success Criteria
- Probe accuracy meaningfully above chance on held-out HC3
- Statistically significant AI-likeness shift under steering
- Clear documentation of layer effects and limitations
