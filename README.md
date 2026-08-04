# Movie Review Sentiment Analysis

## Overview

This project is a Machine Learning application that classifies movie reviews as **Positive** or **Negative** using **Natural Language Processing (NLP)** techniques. The review text is converted into numerical features using **CountVectorizer**, and a **Multinomial Naive Bayes** classifier is trained to predict the sentiment.

---

## Problem Statement

Predict whether a movie review expresses a **positive** or **negative** sentiment based on the review text.

---

## Dataset

- **Source:** Kaggle
- **Dataset Name:** IMDB Movie Reviews Dataset
- **Number of Records:** 50,000
- **Features:**
  - `review` – Movie review text
  - `sentiment` – Positive or Negative

---

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn

---

## Machine Learning Concepts

- Data Loading
- Data Exploration
- Data Visualization
- Text Preprocessing
- CountVectorizer
- Train-Test Split
- Multinomial Naive Bayes
- Accuracy Evaluation
- Sentiment Prediction

---

## Data Visualization

The project includes:

- Sentiment Distribution
- Review Length Distribution

---

## Model Used

- **Algorithm:** Multinomial Naive Bayes
- **Feature Extraction:** CountVectorizer

---

## Model Performance

- **Accuracy:** **84.87%**

---

## Sample Prediction

**Input Review**

```
This movie is absolutely amazing. I loved it!
```

**Predicted Sentiment**

```
Positive
```

---

## Project Structure

```
Movie-Review-Sentiment-Analysis/
│── movie_review.py
│── review.csv
│── README.md
```

---

## Future Improvements

- TF-IDF Vectorization
- Confusion Matrix
- Precision, Recall, and F1 Score
- Hyperparameter Tuning
- Web Interface using Streamlit or Flask

---

## Author

**Pavani Mamillapalli**

Second Year AIML Student

Learning Machine Learning by building practical projects.
