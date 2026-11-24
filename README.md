# ReviewSense

Transformer-Based Multi-Class Classification of Student Reviews

## Overview

ReviewSense uses a fine-tuned DistilBERT model to classify written student reviews into five categories: awesome, good, average, poor, and awful.

## Dataset

The project uses `feedback student.csv` with the columns:
- `Id`: professor identifier
- `comment`: student-written review
- `quality`: target class

The `Id` column is not used as a model input. The model learns from review text.

## Model

DistilBERT (`distilbert-base-uncased`) is fine-tuned for five-class sequence classification.

## Project workflow

1. Load and clean the dataset.
2. Remove duplicate and empty review records.
3. Encode the five target labels.
4. Create stratified training and test sets.
5. Tokenize reviews using the DistilBERT tokenizer.
6. Fine-tune the pretrained Transformer.
7. Evaluate using accuracy, macro F1-score, classification report, and confusion matrix.
8. Use the trained model through a Streamlit application.

## Setup

Place `feedback student.csv` in the project root.

```bash
python -m venv venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

Launch the application:

```bash
streamlit run app.py
```

## Evaluation

The test set is stratified to preserve class proportions. Report the actual accuracy and macro F1-score printed after training. Do not claim a result before running the model.

## Limitations

The dataset contains professor reviews and their existing quality labels. The model predicts those dataset categories; it does not independently verify teaching quality or establish objective facts about a professor. Results depend on the dataset's label quality and size.
