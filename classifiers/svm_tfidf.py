from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib


class SVMTFIDF:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = SVC(kernel='linear', probability=True)

    def train(self, X_train, y_train):
        X_train = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test=None, y_test=None):
        if X_test is None or y_test is None:
            print("No test data provided. Skipping evaluation.")
            return None

        X_test = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        # Calculate precision, recall, and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return accuracy, precision, recall, f1

    def save(self, model_path):
        joblib.dump(self.vectorizer, f'{model_path}_vectorizer.pkl')
        joblib.dump(self.model, f'{model_path}.pkl')
        print("Model saved.")