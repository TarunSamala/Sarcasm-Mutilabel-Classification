# Sarcasm Detection in Tweets

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive NLP project for detecting sarcasm in tweets using multiple deep learning approaches with pre-trained word embeddings.

## Features

- **Multiple Model Architectures**:
  - Bi-LSTM with GloVe embeddings
  - Bi-GRU with FastText embeddings
  - Random Forest with averaged word vectors
  - Bert on Specific Env for version conflict

- **Advanced Text Processing**:
  - Special handling for social media text (URLs, mentions, hashtags)
  - Dynamic sequence padding
  - Comprehensive text cleaning pipeline

- **Visual Analytics**:
  - Correlation heatmap of sarcasm types
  - Training/validation performance curves
  - Model comparison metrics

## Dataset

Dataset we use:
1. `train.En.csv` - Twitter dataset with multiple sarcasm type labels

Dataset columns:
- `tweet`: Original sarcastic tweet
- `rephrase`: Non-sarcastic version
- Binary labels for sarcasm types:
  - `sarcasm`, `irony`, `satire`
  - `understatement`, `overstatement`
  - `rhetorical_question`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/TarunSamala/Sarcasm-Mutilabel-Classification.git
cd sarcasm-detection