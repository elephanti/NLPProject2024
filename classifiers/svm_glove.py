import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

nltk.download('punkt')


class SVMGlove:
    def __init__(self, glove_file):
        self.embeddings_index = self.load_glove_embeddings(glove_file)
        self.vectorizer = None
        self.model = SVC(kernel='linear', probability=True)

    def load_glove_embeddings(self, glove_file):
        embeddings_index = {}
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index

    def get_embedding(self, word, dim=100):
        return self.embeddings_index.get(word, np.zeros(dim))

    def get_text_embedding(self, text, dim=100):
        words = [word.lower() for word in word_tokenize(text)]
        word_embeddings = [self.get_embedding(word, dim) for word in words]
        return np.mean(word_embeddings, axis=0)

    def prepare_data(self, texts):
        return np.vstack([self.get_text_embedding(text) for text in texts])

    def train(self, X_train, y_train):
        X_train = self.prepare_data(X_train)
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test=None, y_test=None):
        if X_test is None or y_test is None:
            print("No test data provided. Skipping evaluation.")
            return None

        X_test = self.prepare_data(X_test)
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
        joblib.dump(self.model, f'{model_path}.pkl')
        print("Model saved.")