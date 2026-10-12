# IMDB Movie Review Sentiment Classifier

A binary sentiment classifier (positive/negative) built on the IMDB movie review dataset, using Word2Vec word embeddings feeding into a stacked LSTM network.

## Results

- **84.69% accuracy** on a 15,000-review held-out test set
- Trained on 35,000 reviews, tested on 15,000 (70/30 split)
- Word2Vec embeddings trained on the training split only, to prevent test-data leakage into the vector space

## Project Overview

This project explores the full pipeline for sequence-based sentiment classification:

1. **Clean & tokenize** 50,000 IMDB reviews (strip HTML tags, lowercase, normalize punctuation spacing)
2. **Split** into train/test *before* training embeddings, so the model's test performance reflects genuinely unseen data
3. **Train Word2Vec** (CBOW(sg=0), 100-dimensional vectors) on the training tokens only
4. **Build an embedding matrix** mapping each vocabulary word to its learned vector
5. **Feed into a two-layer LSTM** (64 units each) with the embeddings left unfrozen (`trainable=True`), so they continue to specialize for sentiment during training
6. **Evaluate** on the held-out test set and sanity-check predictions on a few hand-written example reviews

See [`sentiment_analysis_lstm.ipynb`](sentiment_analysis_lstm.ipynb) for the full walkthrough with code, outputs, and commentary.

## Architecture

| Layer | Output Shape | Params |
|---|---|---|
| Embedding (Word2Vec, 100-dim) | (200, 100) | 3,473,700 |
| LSTM (64 units, returns sequences) | (200, 64) | 42,240 |
| LSTM (64 units) | (64,) | 33,024 |
| Dropout (0.15) | (64,) | 0 |
| Dense (2, softmax) | (2,) | 130 |

**Total params:** 3,549,094, all trainable (embeddings are unfrozen in this version).

## Dataset

[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (Kaggle). Not included in this repo — download `IMDB Dataset.csv` from the link above and place it in the project root before running the notebook.

## Setup

```bash
pip install -r requirements.txt
```

Then open `sentiment_analysis_lstm.ipynb` in Jupyter or Colab. Make sure `data_cleaner.py` and `IMDB Dataset.csv` are in the same directory as the notebook.

## Repo Structure

```
.
├── sentiment_analysis_lstm.ipynb   # main notebook: data prep, training, evaluation
├── data_cleaner.py                 # text cleaning/tokenizing helper used by the notebook
├── requirements.txt
└── README.md
```

## Example Predictions
These example reviews are not included in the 50k IMDb dataset. These were made by me for testing/validation.

| Review | Prediction | P(positive) |
|---|---|---|
| "This movie was a complete waste of time, terrible acting and a boring plot." | Negative | 0.01 |
| "Ehh, I thought it was just alright, nothing special" | Negative | 0.12 |
| "I absolutely loved it!" | Positive | 0.98 |
| "This movie was good." | Positive | 0.68 |
| "This movie was bad." | Negative | 0.16 |

Worth noting: the model has no "neutral" class, it's strictly binary, so an ambiguous/lukewarm review like the second example gets forced into whichever side scores higher, here, correctly leaning negative, but with much lower confidence than the clearly positive/negative examples.

## Possible Next Steps
This project was built as part of my coursework for Hood College's MS in Computer Science (Data Science & AI concentration). Possible next steps I want to explore:
- Compare frozen vs. unfrozen embeddings directly (this version uses unfrozen/`trainable=True`)
- Try a bidirectional LSTM or attention mechanism
- Add a confusion matrix / precision-recall breakdown by class
- Tune `sequence_length` based on the actual review-length distribution rather than a fixed guess
