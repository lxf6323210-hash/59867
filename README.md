# Visual Caption Error Detection

## Overview
This project detects incorrect image captions using BLIP, Sentence-BERT, and Qwen2.5-VL.

It generates image captions, introduces artificial caption errors, and checks whether the captions match the actual images.

## Technologies
- Python 3.12
- PyTorch
- BLIP
- Sentence-BERT
- Qwen2.5-VL
- CUDA
- Matplotlib

## Project Structure
```text
59867/
├── images/
├── results/
├── caption_generator.py
├── semantic_similarity.py
├── visual_verifier.py
├── fault_injection.py
├── evaluation.py
├── main.py
├── metrics.py
└── visualize.py
```

## Installation
Install the required packages:

```bash
pip install torch torchvision torchaudio
pip install transformers sentence-transformers accelerate
pip install pillow requests matplotlib numpy pandas qwen-vl-utils
```

## How to Run

Run the main experiment:

```bash
python main.py
```

Calculate evaluation metrics:

```bash
python metrics.py
```

Generate comparison charts:

```bash
python visualize.py
```

## Results
Experimental results are saved in the `results/` folder.

- `experiment_results.csv`: Experiment records
- `metrics_summary.csv`: Evaluation metrics
- `graphs/`: Performance and coverage charts

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- False Positive Rate
- Coverage

## Project Goal
The goal is to investigate how independent vision-language models can help detect incorrect image captions and compare their performance with text-based semantic similarity methods.
