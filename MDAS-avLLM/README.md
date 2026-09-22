# MDAS: Modality-Decoupled Activation Steering for Hallucination Reduction in Audio-Visual Large Language Models

This is the official code for our ICASSP 2027 submission: **Modality-Decoupled Activation Steering for Hallucination Reduction in Audio-Visual Large Language Models**.

## Overview

MDAS reduces hallucinations in audio-visual large language models (avLLMs) by learning **modality-specific** steering vectors for visual- and audio-induced hallucinations separately, and applying them with layer-specific weighted addition at inference time.

Key ideas:
1. **Reference-free pair construction**: Use SelfCheckGPT scores to identify pseudo-positive/negative continuations without human annotations or reference captions.
2. **Modality-decoupled directions**: Extract separate visual and audio steering vectors via pairwise PCA on contrastive activation pairs.
3. **Layer-specific weighted addition**: Inject the two directions at different layers with independently calibrated strengths, optionally with input-dependent (dynamic) weighting based on a condition vector.

## Model

All experiments use **Qwen2.5-Omni-3B** (`Qwen/Qwen2.5-Omni-3B`).

## Directory Structure

```
MDAS-avLLM/
├── avhbench/                    # AVHBench (question-answering) pipeline
│   ├── step1_selfcheck_sample.py    # Sample K responses per video under audio/visual prompts
│   ├── step2_selfcheck_score.py      # Compute SelfCheckGPT hallucination rate with DeBERTa NLI
│   ├── step3_build_pairs.py        # Select pseudo-positive/negative pairs by consistency score
│   ├── step4_train_steering.py      # Pairwise PCA to extract visual/audio steering vectors
│   ├── step5a_build_condition_pairs.py  # Build AVH/VAH pairs for condition vector
│   ├── step5b_train_condition.py   # Train condition vector for input-dependent weighting
│   ├── step6_infer_dynamic.py      # Dynamic inference: condition-based w_a/w_v fusion
│   └── step7_search_layers.py     # Search optimal 4-layer window and strength
│
├── dvdbench/                    # DVDBench (open-ended description) pipeline
│   ├── step1_selfcheck_sample.py    # SelfCheck sampling
│   ├── step2_selfcheck_score.py     # SelfCheck scoring
│   ├── step3_build_pairs.py        # Pair construction
│   ├── step4_train_steering.py     # Steering vector training
│   ├── step5_search_and_infer.py    # Layer/strength search + MDAS inference (insertion rate)
│   └── step6_evaluate.py           # Evaluate: speaker acc, insertion rate, temporal IoU, SSC, SRC
│
├── baselines/                   # Baseline methods
│   ├── unsteered.py                # Original model without steering
│   ├── vcd.py                      # Visual Contrastive Decoding (VCD)
│   └── avcd.py                     # Audio-Visual Contrastive Decoding (AVCD)
│
├── ablations/                   # Ablation studies
│   ├── layer_search_multigpu.py    # Multi-GPU layer window search
│   ├── fusion_singlelayer.py       # Single-layer fusion (v=1.0, a=1.0)
│   └── singlelayer.py              # Single-layer single-vector inference
│
├── vectors/                    # Pretrained steering vectors (.svec format)
│   ├── avh_audio.svec             # Audio steering vector for AVHBench
│   ├── avh_visual.svec            # Visual steering vector for AVHBench
│   ├── dvd_audio.svec              # Audio steering vector for DVDBench
│   └── dvd_visual.svec             # Visual steering vector for DVDBench
│
└── README.md
```

## Quick Start

### Requirements

```bash
pip install torch transformers accelerate
pip install sentencepiece
pip install sentence-transformers
# ffmpeg for video/audio processing
```

### AVHBench Pipeline

```bash
cd avhbench

# Step 1: SelfCheck sampling (K=10 responses per prompt per video)
python step1_selfcheck_sample.py

# Step 2: Score with DeBERTa NLI to get hallucination rate Q
python step2_selfcheck_score.py

# Step 3: Build contrastive pairs
python step3_build_pairs.py

# Step 4: Train visual and audio steering vectors (pairwise PCA)
python step4_train_steering.py

# Step 5: Train condition vector (for dynamic weighting)
python step5a_build_condition_pairs.py
python step5b_train_condition.py

# Step 6: Run dynamic inference
python step6_infer_dynamic.py

# Step 7: Search optimal layers and strengths
python step7_search_layers.py
```

### DVDBench Pipeline

```bash
cd dvdbench

python step1_selfcheck_sample.py
python step2_selfcheck_score.py
python step3_build_pairs.py
python step4_train_steering.py
python step5_search_and_infer.py     # Layer search + MDAS inference
python step6_evaluate.py             # Evaluate all metrics
```

### Baselines

```bash
cd baselines
python unsteered.py    # No steering
python vcd.py          # VCD with noisy visual input
python avcd.py         # AVCD with masked visual/audio
```

## Final Configuration

| Benchmark | Visual layers | Visual strength | Audio layers | Audio strength | Weighting |
|-----------|--------------|-----------------|--------------|----------------|-----------|
| AVHBench  | [25,26,27,28] | 2.0 | [24,25,26,27] | 3.0 | Input-dependent (condition vector, τ=0.077) |
| DVDBench  | [18,19,20,21] | 1.0 | [18,19,20,21] | 0.5 | Static |

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhou2027mdas,
  title={Modality-Decoupled Activation Steering for Hallucination Reduction in Audio-Visual Large Language Models},
  author={Zhou, Ziqi and Guo, Yuxin and Jin, Zengrui and Sun, Guangzhi and Zhang, Chao},
  booktitle={ICASSP},
  year={2027}
}
```
