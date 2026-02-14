# Sounds-Like-AI Residual Direction

This project tests whether “sounds like AI” corresponds to a linear direction in the residual stream, using HC3 (human vs AI) and activation steering on a modern small LLM.

## Key Findings
- Linear probe on hidden states separates human vs AI with high accuracy (best layer 12: 97.8%).
- Steering along the mean-difference direction shifts AI-likeness scores upward/downward in the expected direction.
- Effects are suggestive but not statistically significant in this small pilot.

## How to Reproduce
1. Activate environment:
```bash
source .venv/bin/activate
```
2. Run experiments:
```bash
python src/experiment.py
```
3. Run GPT-4.1 judge scoring (requires `OPENAI_API_KEY`):
```bash
python src/judge_ai_likeness.py
```
4. Analyze results:
```bash
python src/analyze_results.py
```

## File Structure
- `src/experiment.py` — data prep, probing, steering, and generation
- `src/judge_ai_likeness.py` — GPT-4.1 AI-likeness ratings
- `src/analyze_results.py` — stats and analysis summary
- `results/` — metrics, plots, and generated outputs
- `planning.md` — planning and motivation
- `REPORT.md` — full research report

See `REPORT.md` for methodology and detailed results.
