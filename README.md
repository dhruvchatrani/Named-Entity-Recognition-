# Advanced Named Entity Recognition Using BiLSTM-CRF and GloVe

This Named Entity Recognition (NER) system uses a Bi-directional LSTM (BiLSTM) coupled with a Conditional Random Field (CRF) layer. The model uses GloVe word embeddings and includes a data augmentation pipeline to improve generalization.

The code compares three specific embedding strategies to see how different handling of pre-trained vectors and unknown words affects performance.

## Features

  * **Architecture:** BiLSTM captures context, while the CRF layer handles sequence tagging constraints.
  * **Embeddings:** Integrates GloVe (Global Vectors for Word Representation).
  * **Data Augmentation:** A pipeline that replaces words with GloVe-based synonyms to increase training data variation.
  * **Experimentation:** Automatically trains three different embedding fine-tuning strategies.
  * **Training Loop:** Includes Early Stopping, Model Checkpointing, and Learning Rate Scheduling.
  * **Visualization:** Generates training loss and F1-score plots automatically.


## Model Architecture & Strategies

The model consists of:

1.  **Embedding Layer:** 100-dimension GloVe vectors.
2.  **BiLSTM Layer:** 2 layers, 200 hidden dimensions.
3.  **Linear Layer:** Maps LSTM output to the tag space.
4.  **CRF Layer:** Decodes the sequence of tags based on transition probabilities (e.g., `I-ORG` cannot follow `B-PER`).

### Embedding Strategies

The script tests three ways to handle embeddings:

| Strategy | Description | Trainable? | Handling Unknowns (`<UNK>`) |
| :--- | :--- | :--- | :--- |
| **Strategy A** | **Frozen GloVe**. Uses pre-trained vectors exactly as they are. | No | Zeros vector |
| **Strategy B** | **Fine-Tuned**. Starts with GloVe, but allows updates during training. | Yes | Random Normal Distribution |
| **Strategy C** | **Fine-Tuned + Mean**. Starts with GloVe, allows updates. | Yes | Mean of all GloVe vectors |

## Results

Training logs on the test set show **Strategy C** performed best.

| Metric | Strategy A (Frozen) | Strategy B (Fine-Tuned) | Strategy C (Mean UNK) |
| :--- | :---: | :---: | :---: |
| **F1 Score** | 0.8579 | 0.8982 | **0.9073** |

Strategy C achieved an F1 score of 0.9073. Initializing unknown tokens with the mean of the embedding space while allowing fine-tuning provided the best results for this dataset.

## Getting Started

### Prerequisites

Make sure Python is installed. You also need the `glove.6B.100d.txt` file in your root directory (available from Stanford NLP).

### Installation

Install the dependencies:

```bash
pip install torch torchcrf tqdm numpy matplotlib
```

### Data Format

The system expects data in standard **CoNLL format** (whitespace separated):

```text
EU   NNP  I-NP  B-ORG
rejects VBZ I-VP O
German JJ   I-NP B-MISC
```

### Running the Training

Run the script directly. It will:

1.  Load and parse data.
2.  Run data augmentation.
3.  Train all three strategies in order.
4.  Save the best model to `best_model.pt`.
5.  Create performance plots.

<!-- end list -->

```bash
python main.py
```

## Tech Stack

  * **Language:** Python
  * **DL Framework:** PyTorch
  * **Sequence Modeling:** `torchcrf`
  * **Visualization:** Matplotlib
  * **Data Processing:** NumPy, TQDM
