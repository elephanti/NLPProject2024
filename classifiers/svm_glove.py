import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

nltk.download('punkt')


class SVM_Glove:
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

    def train(self, train_texts, train_labels):
        X_train = self.prepare_data(train_texts)
        self.model.fit(X_train, train_labels)

    def evaluate(self, test_texts=None, test_labels=None):
        if test_texts is None or test_labels is None:
            print("No test data provided. Skipping evaluation.")
            return None

        X_test = self.prepare_data(test_texts)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(test_labels, y_pred)
        return accuracy

    def save(self, model_path):
        joblib.dump(self.model, f'classifiers/models/{model_path}.pkl')
        print("Model saved.")