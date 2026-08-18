from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Load Dataset
dataset = load_dataset("stanfordnlp/imdb")

train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label"]

test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

#Convert text into TF-IDF vectors

vecotorizer = TfidfVectorizer()

X_train = vecotorizer.fit_transform(train_texts)
X_test = vecotorizer.transform(test_texts)

print("Training Shape" , X_train.shape)
print("Testing Shape" , X_test.shape)

#Create Model

model = LogisticRegression(max_iter=1000)

#Train Model

model.fit(X_train , train_labels)

#Make Predictions

predictions = model.predict(X_test)

#Calculate Accuracy

accuracy = accuracy_score(test_labels,predictions)

print(f"Test Accuracy: {accuracy * 100:.2f}%")