import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import accuracy_score

df=pd.read_csv("review.csv")

print("First 5 Rows")
print(df.head())

print("\nData Information") 
df.info()

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

print("\nColumns")
print(df.columns)

sentiment_count=df["sentiment"].value_counts()
plt.bar(sentiment_count.index,sentiment_count.values,color=["green","red"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.show()

X=df["review"]

y=df["sentiment"]

vectorizer=CountVectorizer()

X=vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)

model=MultinomialNB()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

new_review=["This movie is absolutely amazing. I loved it!"]


new_review_vector=vectorizer.transform(new_review)

prediction=model.predict(new_review_vector)

print("\nInput Review")
print(new_review[0])

print("\nPredicted Sentiment:",prediction[0])