# TinyGPT — A Minimal GPT Model Trained From Scratch

This project is a small experimental GPT-like model implemented entirely in Python using PyTorch.  
The main goal of the project was to understand how transformer-based language models actually work by building one from the ground up.

The model currently trains and runs without errors, but its outputs are still mostly gibberish due to limited training data, a very small model size, and short training time. I plan to keep improving it.

---

## Overview

The project contains two main parts:

1. **A miniature GPT model**  
   - Causal self-attention  
   - Multi-head attention  
   - Positional embeddings  
   - Transformer blocks  
   - Weight tying between the embedding layer and output layer  

2. **A simple command-line chatbot**  
   - Loads the trained checkpoint  
   - Generates responses token by token  
   - Allows basic back-and-forth chat in the terminal  

This setup was mainly built for learning and experimentation.

---

## Dataset

I trained the model on a small subset (20,000 samples) of the **OpenAssistant OASST2** dataset.  
This dataset is designed for instruction-following tasks, but the subset I used is far too small to produce meaningful results.

Because of that, the model currently does not generate coherent text and tends to output random tokens.

---

## Training

The training script performs the following:

- Loads and cleans the dataset  
- Trains a **Byte-Level BPE tokenizer** from scratch  
- Converts text into token sequences  
- Creates fixed-length batches  
- Trains the transformer using next-token prediction  

A sample training command looks like:

```bash
python train.py --max_steps 5000 --batch_size 2 --grad_accum 8
```

Checkpoints are saved under:

```
ckpt/gpt_stepXXXX.pt
```

---

## Chatbot

After training, you can run:

```bash
python chatbot.py
```

This loads the tokenizer and model, then starts an interactive loop.  
Right now, responses are mostly random or incoherent — this is expected with such a tiny model and dataset.

Improving this is part of future plans.

---

## Current Limitations

This project is intentionally small, and comes with several limitations:

- Very small transformer (4 layers, 4 heads, 256 hidden size)  
- Trained on only 20,000 samples  
- Training time is short  
- No large-scale text corpus  
- No special sampling strategies (basic top-k only)  

Because of these constraints, the model cannot produce meaningful text yet.  
The purpose of the project was to learn how everything fits together, not to build a competitive LLM.

---

## Future Work

I plan to improve the project by:

- Training on a much larger dataset  
- Increasing the number of layers, heads, and embedding size  
- Trying better positional encoding methods  
- Adding top-p sampling and repetition penalties  
- Improving the training loop  
- Experimenting with curriculum training  
- Building a simple web interface for the chatbot  

As the model scales up, the output should gradually become more coherent.

---

## Requirements

```
torch
datasets
tokenizers
transformers
```

Install with:

```bash
pip install torch datasets tokenizers transformers
```

---

## Project Structure

```
small-scale-llm/
│
├── newgpt.py                # Training code
├── chat_with_model.py              # CLI chatbot using the trained model
├── train_texts.json            # Model training data
└── README.md
```

---

## Notes

This is a learning project, not a production model.  
The purpose was to write a transformer from scratch, train it on a small dataset, and understand the full pipeline end-to-end.

The model works technically, but the outputs are not meaningful yet — I’m planning to keep improving it over time.

