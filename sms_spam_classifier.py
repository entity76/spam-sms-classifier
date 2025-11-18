import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("sms_spam_dataset.csv")  # we will add this file next
df.columns = ["label", "message"]

# Convert labels to 0/1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

# 3. Convert text → TF-IDF (numeric)
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 5. Predict
predictions = model.predict(X_test_tfidf)

# 6. Evaluate
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# ---- Predict on custom message ----
while True:
    msg = input("\nEnter a message to classify (or type 'exit'): ")

    if msg.lower() == "exit":
        break

    msg_tfidf = vectorizer.transform([msg])
    pred = model.predict(msg_tfidf)[0]

    if pred == 1:
        print("Prediction: SPAM ❌")
    else:
        print("Prediction: HAM ✔")

