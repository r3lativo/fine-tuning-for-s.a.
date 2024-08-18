# Fine-Tuning for Sentiment Analysis

This repository contains a Python notebook demonstrating the fine-tuning of some pre-trained models for sentiment analysis on the Stanford IMDB reviews dataset. The project explores both standard fine-tuning and a parameter-efficient fine-tuning approach using Low-Rank Adaptation (LoRA).

## Project Overview

- **Model_1**: `bert-base-uncased` (a pre-trained BERT model)
- **Model_2**: `llama...` (a pre-trained Llama model) [TODO]
- **Dataset**: Stanford IMDB reviews (for binary sentiment classification)
- **Libraries**: PyTorch, Hugging Face Transformers

## Key Features

- **Tokenization**: Converts raw text into tokens using associated tokenizer.
- **Standard Fine-Tuning**: Updates all model parameters using AdamW optimizer with carefully selected hyperparameters.
- **LoRA Fine-Tuning**: Efficiently fine-tunes BERT by injecting trainable low-rank matrices into modules of the Transformer layers.

## Files

- `bert_fine_tuning.ipynb`: Main notebook containing all the code, explanations, and outputs.
- `requirements.txt`: List of required packages and dependencies.

