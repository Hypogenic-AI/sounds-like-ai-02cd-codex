# Literature Review

## Research Area Overview
This project targets inference-time activation steering in the residual stream to manipulate output style and semantics, with an emphasis on identifying a “sounds like AI” direction. Recent work shows that linear directions in residual-stream activations can reliably steer behaviors (CAA, Selective Steering), but reliability and interpretability depend on context, layer selection, and representation sparsity (SVF, SRS). Parallel work on multi-layer SAEs suggests latent features are distributed across layers, which affects how a style direction should be extracted and applied. Finally, AI-vs-human text datasets (HC3, HC3 Plus) provide supervision for “sounds like AI” detection and evaluation.

## Key Papers

### Paper 1: Steering Llama 2 via Contrastive Activation Addition
- **Authors**: Panickssery et al.
- **Year**: 2024
- **Source**: arXiv 2312.06681
- **Key Contribution**: Introduces CAA—steering vectors computed as mean differences of residual-stream activations from contrastive prompt pairs, applied at inference by adding vectors across tokens.
- **Methodology**: Contrastive multiple-choice datasets; mean-difference vectors at answer-token positions; apply to all post-prompt tokens.
- **Datasets Used**: Anthropic model-written evals, sycophancy datasets, TruthfulQA, MMLU; custom hallucination/refusal datasets.
- **Results**: CAA steers multiple behaviors, generalizes to open-ended generation, minimal capability loss, and can combine with prompting/finetuning.
- **Code Available**: Yes (CAA repo).
- **Relevance to Our Research**: Baseline method for extracting residual-stream directions; provides recipe for steering vectors and evaluation setup.

### Paper 2: Selective Steering: Norm-Preserving Control Through Discriminative Layer Selection
- **Authors**: Dang & Ngo
- **Year**: 2026
- **Source**: arXiv 2601.19375
- **Key Contribution**: Norm-preserving rotation + discriminative layer selection yields more stable steering and better controllability than uniform layer edits.
- **Methodology**: Difference-in-means direction, discriminative layers (opposite-signed mean projections), norm-preserving rotation in a 2D plane.
- **Datasets Used**: AdvBench, Alpaca, tinyBenchmarks (tinyMMLU, tinyTruthfulQA, etc.).
- **Results**: Higher attack success rates with minimal coherence loss vs prior steering methods.
- **Code Available**: Yes (Selective Steering repo).
- **Relevance to Our Research**: Important for deciding where to steer in the residual stream to avoid style collapse.

### Paper 3: Improved Representation Steering for Language Models (RePS)
- **Authors**: Wu et al.
- **Year**: 2025
- **Source**: arXiv 2505.20809
- **Key Contribution**: RePS (reference-free, bidirectional preference steering) improves representation steering vs LM objective; narrows gap with prompting.
- **Methodology**: Preference-optimization objective (SimPO-derived) for steering vectors/LoRA/ReFT; evaluated on AXBENCH.
- **Datasets Used**: AXBENCH steering benchmark; Alpaca-Eval, Dolly, GSM8K, etc.
- **Results**: Better steering/suppression on Gemma models; robust to prompt jailbreaking.
- **Code Available**: AXBENCH (github.com/stanfordnlp/axbench).
- **Relevance to Our Research**: Suggests training objectives for steering vectors that may improve “sounds like AI” control.

### Paper 4: Steering Vector Fields for Context-Aware Inference-Time Control in LLMs
- **Authors**: Li et al.
- **Year**: 2026
- **Source**: arXiv 2602.01654
- **Key Contribution**: Context-dependent steering via vector fields; avoids failures of global vectors in long-form or multi-attribute settings.
- **Methodology**: Learn differentiable concept boundary; steer by local gradient; align representations across layers.
- **Datasets Used**: Model-Written-Evals; multiple steering tasks across LLMs.
- **Results**: Stronger, more reliable control than static vectors.
- **Code Available**: Not specified in abstract.
- **Relevance to Our Research**: “Sounds like AI” may vary with context; vector-field approach could improve reliability.

### Paper 5: On the Identifiability of Steering Vectors in LLMs
- **Authors**: Venkatesh & Kurapath
- **Year**: 2026
- **Source**: arXiv 2602.06801
- **Key Contribution**: Proves steering vectors are non-identifiable without structural assumptions; empirically shows orthogonal perturbations can be equally effective.
- **Methodology**: Formal analysis of equivalence classes; empirical validation across traits and models.
- **Datasets Used**: Multiple traits; unspecified in abstract.
- **Results**: Identifiability can be recovered with sparsity, independence, multi-environment validation, or cross-layer consistency.
- **Code Available**: Not specified in abstract.
- **Relevance to Our Research**: Warns against over-interpreting a single “sounds like AI” direction; suggests constraints for stability.

### Paper 6: Endogenous Resistance to Activation Steering in Language Models
- **Authors**: McKenzie et al.
- **Year**: 2026
- **Source**: arXiv 2602.06941
- **Key Contribution**: Large models can resist misaligned steering, self-correcting mid-generation (ESR).
- **Methodology**: SAE-latent steering with off-topic latents; judge-based multi-attempt scoring; ablation of ESR-related latents.
- **Datasets Used**: Curated “explain how” prompts; SAE latents from Goodfire/GemmaScope.
- **Results**: Llama-3.3-70B shows ESR; 26 latents causally linked; ESR increased by prompting and finetuning.
- **Code Available**: Yes (endogenous-steering-resistance repo).
- **Relevance to Our Research**: Potential failure mode for “sounds like AI” steering if models resist style perturbations.

### Paper 7: Interpretable LLM Guardrails via Sparse Representation Steering
- **Authors**: He et al.
- **Year**: 2025
- **Source**: arXiv 2503.16851
- **Key Contribution**: Sparse Representation Steering (SRS) uses SAE space + KL divergence to pick monosemantic features; supports multi-attribute control.
- **Methodology**: Project to SAE latent space; identify features via bidirectional KL; compose sparse vectors.
- **Datasets Used**: Gemma-2 models; safety/fairness/truthfulness tasks.
- **Results**: Better controllability and quality vs dense steering; robust multi-attribute control.
- **Code Available**: Stated in paper.
- **Relevance to Our Research**: SAE-based sparse directions may isolate “AI-like” style more cleanly.

### Paper 8: Residual Stream Analysis with Multi-Layer SAEs
- **Authors**: Lawson et al.
- **Year**: 2025
- **Source**: arXiv 2409.04185 (ICLR 2025)
- **Key Contribution**: Multi-layer SAE (MLSAE) analyzes residual stream across layers; latents can shift activation layer-by-layer.
- **Methodology**: Single SAE trained on activations from all layers; analyze latent distributions over layers.
- **Datasets Used**: Pythia models; token-level analyses.
- **Results**: Latent activation layer depends on token/prompt; more multi-layer activation with larger models.
- **Code Available**: Yes (mlsae repo).
- **Relevance to Our Research**: Suggests “sounds like AI” direction may not be fixed to one layer.

### Paper 9: HC3: Human ChatGPT Comparison Corpus
- **Authors**: Guo et al.
- **Year**: 2023
- **Source**: arXiv 2301.07597
- **Key Contribution**: 40k+ question-answer pairs with human and ChatGPT responses; detection baselines.
- **Methodology**: Collect human QA from datasets; prompt ChatGPT for responses; train detectors.
- **Datasets Used**: ELI5, WikiQA, medical, finance, etc.
- **Results**: Provides strong dataset for AI-vs-human detection.
- **Code Available**: Yes (Hello-SimpleAI repo).
- **Relevance to Our Research**: Direct supervision for “sounds like AI” vs human style.

### Paper 10: HC3 Plus: A Semantic-Invariant Human ChatGPT Comparison Corpus
- **Authors**: Su et al.
- **Year**: 2024
- **Source**: arXiv 2309.02731
- **Key Contribution**: Adds semantic-invariant tasks (summarization/translation/paraphrasing) that are harder for detectors.
- **Methodology**: Build HC3-SI; finetune instruction-following detector (Tk-instruct).
- **Datasets Used**: CNN/DailyMail, XSum, LCSTS, WMT, etc.
- **Results**: Detection is harder for semantic-invariant tasks; HC3 Plus improves coverage.
- **Code Available**: Yes (HC3-Plus repo).
- **Relevance to Our Research**: Tests robustness of “AI-like” style detection beyond QA.

## Common Methodologies
- **Mean-difference steering vectors**: Contrastive prompts (CAA, SRS) to derive a direction in residual stream.
- **Layer selection**: Identify discriminative layers where representation separates (Selective Steering).
- **Context-dependent steering**: Vector fields using local gradients (SVF).
- **Sparse feature steering**: SAE latent space for interpretable features (SRS, MLSAE).

## Standard Baselines
- **Activation Addition / CAA**: Add a single vector at a chosen layer.
- **Directional Ablation**: Remove or project out a direction.
- **Angular Steering**: Rotate in a 2D plane (with/without norm preservation).
- **Prompting / system prompts**: Non-interventional baseline.

## Evaluation Metrics
- **Steering success**: Task-specific accuracy, GPT/LLM-judge ratings, or classifier accuracy.
- **Quality preservation**: Perplexity, repetition, compression ratio, fluency.
- **Robustness**: Benchmark accuracy (MMLU/TruthfulQA variants), resistance to jailbreaks.
- **Detection performance**: Accuracy/precision/recall for AI-vs-human classification on HC3/HC3 Plus.

## Datasets in the Literature
- **HC3**: Human vs ChatGPT QA responses.
- **HC3 Plus**: Adds summarization/translation/paraphrasing.
- **AdvBench / Alpaca**: Harmful vs harmless prompts (steering calibration).
- **AXBENCH**: Steering benchmark for concepts and suppression.

## Gaps and Opportunities
- **Identifiability limits**: A “sounds like AI” direction may not be unique; need structural constraints (sparsity, cross-layer consistency).
- **Context dependence**: Static vectors may fail in long-form generation; vector-field steering could improve stability.
- **Layer dynamics**: Multi-layer latent dynamics suggest the style direction may shift by layer or token.
- **Evaluation**: “Sounds like AI” should be measured with human/AI detection and stylistic attributes, not just task success.

## Recommendations for Our Experiment
- **Recommended datasets**: HC3 (primary), HC3 Plus (robustness to semantic-invariant tasks).
- **Recommended baselines**: CAA, Selective Steering, SRS (SAE-based), SVF if feasible.
- **Recommended metrics**: Detection accuracy (human vs AI), steering success rate, perplexity, repetition, human/LLM-judge quality ratings.
- **Methodological considerations**: Use discriminative layer selection; test for ESR/self-correction; compare sparse vs dense steering for controllability.
