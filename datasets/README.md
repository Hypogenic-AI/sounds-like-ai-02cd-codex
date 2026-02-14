# Downloaded Datasets

This directory contains datasets for the research project. Data files are not committed to git due to size. Follow the download instructions below to reproduce.

## Dataset 1: HC3 (Human ChatGPT Comparison Corpus)

### Overview
- **Source**: HuggingFace dataset `Hello-SimpleAI/HC3`
- **Size**: 24,322 QA items (train split only), ~42 MB on disk
- **Format**: HuggingFace Dataset saved with `save_to_disk`
- **Task**: Human vs ChatGPT detection / style analysis
- **Splits**: train (24,322)
- **Schema**: `id`, `question`, `human_answers`, `chatgpt_answers`, `source`
- **License**: See dataset card on HuggingFace

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

ds = load_dataset("Hello-SimpleAI/HC3", "all", trust_remote_code=True)
ds.save_to_disk("datasets/HC3")
```

### Loading the Dataset
```python
from datasets import load_from_disk

ds = load_from_disk("datasets/HC3")
print(ds["train"][0])
```

### Sample Data
See `datasets/HC3/samples/sample.json` and `datasets/HC3/samples/summary.json`.

### Notes
- The dataset loader uses a custom script; pass `trust_remote_code=True`.

## Dataset 2: HC3 Plus (Semantic-Invariant Human ChatGPT Comparison Corpus)

### Overview
- **Source**: GitHub repo `suu990901/chatgpt-comparison-detection-HC3-Plus`
- **Size**: ~73 MB for `data/en` and `data/zh`
- **Format**: JSONL files
- **Task**: Human vs ChatGPT detection on QA + semantic-invariant tasks (summarization, translation, paraphrasing)
- **Splits**: train/val/test for `hc3_QA` and `hc3_si` (English + Chinese)
- **License**: See paper and repo

### Download Instructions

**Using GitHub (recommended):**
```bash
git clone https://github.com/suu990901/chatgpt-comparison-detection-HC3-Plus code/hc3-plus
cp -r code/hc3-plus/data datasets/HC3-Plus/
```

### Loading the Dataset
```python
import json

with open("datasets/HC3-Plus/data/en/train.jsonl") as f:
    for _ in range(3):
        print(json.loads(next(f)))
```

### Sample Data
See `datasets/HC3-Plus/samples/en_train_head3.jsonl` and `datasets/HC3-Plus/samples/zh_train_head3.jsonl`.

### Notes
- `data/en` and `data/zh` include QA and semantic-invariant (SI) splits.
