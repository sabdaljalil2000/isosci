---
license: cc-by-4.0
task_categories:
  - question-answering
  - text-classification
language:
  - en
tags:
  - science
  - reasoning
  - benchmark
  - isomorphic
  - physics
  - chemistry
  - biology
  - earth-science
size_categories:
  - n<1K
---

# IsoSci-150: Isomorphic Cross-Domain Science Problem Pairs

## Overview

IsoSci-150 is a benchmark of **150 isomorphic problem pairs** designed to cleanly separate 
reasoning ability from domain knowledge retrieval in large language model evaluation.

Each pair consists of two problems:
- **Source problem**: from one scientific domain (e.g., physics)
- **Target problem**: from a different domain (e.g., chemistry) with identical logical structure

The key property: if a model solves the source but fails the target, the bottleneck is 
**domain knowledge**, not **reasoning ability**. This enables direct attribution of 
performance gaps.

## Motivation

Standard science benchmarks (GPQA, SciBench, MMLU-STEM) confound reasoning and knowledge:
it is impossible to determine whether a model fails because it cannot reason or because it 
lacks the relevant domain fact. IsoSci-150 solves this by holding reasoning structure constant.

## Dataset Structure

```
isosci150/
├── train.json    (80 pairs — for few-shot / in-context learning)
├── test.json     (70 pairs — held-out evaluation)
```

Each entry has the following fields:

```json
{
  "pair_id": "uuid",
  "mapping": "physics_to_chemistry",
  "source_domain": "physics",
  "target_domain": "chemistry",
  "source": {
    "question": "A 2.0 mol sample of ideal gas...",
    "answer": "0.996 atm",
    "structure_type": "formula_recall_and_substitute",
    "domain": "physics"
  },
  "target": {
    "question": "What is the pH of a 0.1 M acetic acid solution...",
    "answer": "2.87",
    "structure_type": "formula_recall_and_substitute",
    "domain": "chemistry"
  },
  "verification_status": "auto_accepted",
  "verification_scores": { ... }
}
```

## Domain Mappings

| Mapping | N pairs |
|---------|---------|
| Physics → Chemistry | 25 |
| Physics → Biology | 25 |
| Physics → Earth Science | 25 |
| Chemistry → Biology | 25 |
| Chemistry → Earth Science | 25 |
| Biology → Earth Science | 25 |

## Evaluation

Use the provided `evaluation/eval_script.py`:

```bash
python evaluation/eval_script.py \
    --model openai/gpt-4o \
    --split test \
    --api-key YOUR_OPENROUTER_KEY
```

## Metrics

- **Source accuracy**: accuracy on source domain problems
- **Target accuracy**: accuracy on target domain problems  
- **Domain gap δ**: source_accuracy − target_accuracy
- **Knowledge-dependent gain ratio**: fraction of gains that are domain-specific

## Citation

```bibtex
@dataset{isosci150_2025,
  title={IsoSci-150: Isomorphic Cross-Domain Science Problem Pairs for 
         Evaluating Reasoning vs. Knowledge Retrieval in LLMs},
  author={[Author names]},
  year={2025},
  url={https://huggingface.co/datasets/[your-org]/isosci-150},
  license={CC-BY-4.0}
}
```

## License

CC-BY-4.0. Please cite the associated NeurIPS 2025 paper if you use this dataset.
