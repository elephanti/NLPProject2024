from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib


class SVM_TFIDF:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = SVC(kernel='linear', probability=True)

    def train(self, train_texts, train_labels):
        X_train = self.vectorizer.fit_transform(train_texts)
        self.model.fit(X_train, train_labels)

    def evaluate(self, test_texts=None, test_labels=None):
        if test_texts is None or test_labels is None:
            print("No test data provided. Skipping evaluation.")
            return None

        X_test = self.vectorizer.transform(test_texts)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(test_labels, y_pred)
        return accuracy

    def save(self, model_path):
        joblib.dump(self.vectorizer, f'classifiers/models/{model_path}_vectorizer.pkl')
        joblib.dump(self.model, f'classifiers/models/{model_path}.pkl')
        print("Model saved.")