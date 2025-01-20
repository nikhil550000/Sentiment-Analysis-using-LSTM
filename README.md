# Sentiment Analysis with LSTM in PyTorch

## Overview
This repository demonstrates how to build, preprocess, and train a Long Short-Term Memory (LSTM) model for sentiment analysis using PyTorch. The model processes textual data and predicts sentiment classes (e.g., positive, negative, or neutral). The implementation includes data preprocessing, feature extraction, model definition, and training.

---

## Features
- Preprocess raw text data by:
  - Removing special characters and stopwords.
  - Tokenizing and normalizing the text.
  - Applying TF-IDF vectorization.
- Encode target labels (sentiment) using label encoding.
- Build a custom LSTM using PyTorch for sentiment classification.
- Train the model on batched data using PyTorch's `Dataset` and `DataLoader` utilities.
- Visualize data distribution and word frequencies.

---

## Dataset
The dataset used for this project is available at:
[Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset?select=test.csv).

### Dataset Structure
The dataset includes columns such as:
- **text**: The text data for sentiment analysis.
- **sentiment**: Target labels indicating sentiment (e.g., positive, negative).

---

## Requirements
To run the code, ensure the following dependencies are installed:

```bash
numpy
pandas
matplotlib
seaborn
nltk
scikit-learn
pytorch
torchvision
torchtext
```

# Model Results

## Data Visualizations:

### Word Frequency Distribution:
![Screenshot from 2025-01-21 01-16-59](https://github.com/user-attachments/assets/4a875d5e-8f18-4469-8000-dc4a3ef4aeb5)

### Sentiment Distribution (Positive, Negative, Neutral):
![Screenshot from 2025-01-21 01-16-40](https://github.com/user-attachments/assets/6412e1a0-fef2-438c-bec1-de869c5fa6c5)

A pie chart showing the distribution of sentiments in the dataset, indicating the proportions of positive, negative, and neutral sentiments.

###  Histogram Plot:
![Screenshot from 2025-01-21 01-16-23](https://github.com/user-attachments/assets/5fa7a2f6-3cd5-43ec-a103-469e8a872da9)

An additional histogram that visualizes another aspect of the data, providing insights into its distribution.

## Model Performance:
The model achieved a test accuracy of 80.77%, demonstrating good performance in predicting sentiments from textual data.




## Future Improvements
- Add a validation loop during training.
- Explore advanced text embeddings such as Word2Vec, GloVe, or BERT.
- Deploy the model as an API using Flask or FastAPI.
- Optimize hyperparameters using grid search or random search.

## License
This project is licensed under the MIT License.

## Acknowledgments
Special thanks to Kaggle for providing the dataset and to the PyTorch community for their extensive documentation.
