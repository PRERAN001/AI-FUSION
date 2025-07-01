import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_excel("actual_data.xlsx")
unique_combinations = data['Column2'].unique()
label_mapping = {label: idx for idx, label in enumerate(unique_combinations)}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}  
data['Column2'] = data['Column2'].map(label_mapping)
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=5000)
data["Column3"] = data["Column3"].fillna("").astype(str)
X = vectorizer.fit_transform(data["Column3"])
y = data["Column2"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
my_input = ["solve 3x+56y"]
my_input_tfidf = vectorizer.transform(my_input)
prediction = model.predict(my_input_tfidf) 
predicted_category = reverse_label_mapping[prediction[0]]
print("Predicted label:", predicted_category)
import joblib
joblib.dump(model, 'logistic_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(reverse_label_mapping, 'reverse_label_mapping.pkl')
