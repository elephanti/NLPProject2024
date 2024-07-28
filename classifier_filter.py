import joblib
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from nltk.tokenize import word_tokenize


class DatasetFilter:
    def __init__(self, classifier_name, model_path, glove_file=None, label_encoder_path=None):
        self.classifier_name = classifier_name
        self.model_path = model_path
        self.glove_file = glove_file
        self.vectorizer = None
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.load_model()
        if label_encoder_path:
            self.label_encoder = joblib.load(label_encoder_path)

    def load_model(self):
        if self.classifier_name == 'bert':
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model.to(self.device)
        elif self.classifier_name == 'svm_tfidf':
            self.model = joblib.load(f'{self.model_path}.pkl')
            self.vectorizer = joblib.load(f'{self.model_path}_vectorizer.pkl')
        elif self.classifier_name == 'svm_glove':
            self.model = joblib.load(f'{self.model_path}.pkl')
            self.glove_embeddings = self.load_glove_embeddings(self.glove_file)
        elif self.classifier_name == 'lstm_glove':
            self.model = torch.load(f'{self.model_path}.pth', map_location=self.device)
            self.model.to(self.device)
            self.model.eval()
            self.glove_embeddings = joblib.load(f'{self.model_path}_glove.pkl')

    def load_glove_embeddings(self, glove_file):
        embeddings_index = {}
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index

    def get_embedding(self, text):
        tokens = word_tokenize(text)
        embeddings = np.zeros(100)
        for word in tokens:
            embeddings += self.glove_embeddings.get(word.lower(), np.zeros(100))
        embeddings = embeddings / max(1, len(tokens))  # Average the embeddings
        return embeddings

    def filter_dataset(self, dataset_path, threshold, output_path):
        data = pd.read_csv(dataset_path)
        texts = data['text']
        true_labels = data['label']
        filtered_rows = []

        if self.classifier_name == 'bert':
            for idx, (text, true_label) in enumerate(zip(texts, true_labels)):
                inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(
                    self.device)
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                max_prob, pred_label = torch.max(probs, dim=1)
                pred_label = self.label_encoder.inverse_transform([pred_label.item()])[0]
                if pred_label == true_label and max_prob.item() >= threshold:
                    row = data.iloc[idx].copy()
                    row['predicted_label'] = pred_label
                    row['probability'] = max_prob.item()
                    filtered_rows.append(row)
        elif self.classifier_name == 'svm_tfidf':
            X = self.vectorizer.transform(texts)
            probs = self.model.predict_proba(X)
            preds = self.model.predict(X)
            for idx, (prob, pred, true_label) in enumerate(zip(probs, preds, true_labels)):
                max_prob = np.max(prob)
                pred_label = self.label_encoder.inverse_transform([pred])[0]
                if pred_label == true_label and max_prob >= threshold:
                    row = data.iloc[idx].copy()
                    row['predicted_label'] = pred_label
                    row['probability'] = max_prob
                    filtered_rows.append(row)
        elif self.classifier_name == 'svm_glove':
            embeddings = np.array([self.get_embedding(text) for text in texts])
            if embeddings.shape[1] != 100:
                raise ValueError(f"X has {embeddings.shape[1]} features, but SVC is expecting 100 features as input.")
            probs = self.model.predict_proba(embeddings)
            preds = self.model.predict(embeddings)
            for idx, (prob, pred, true_label) in enumerate(zip(probs, preds, true_labels)):
                max_prob = np.max(prob)
                pred_label = self.label_encoder.inverse_transform([pred])[0]
                if pred_label == true_label and max_prob >= threshold:
                    row = data.iloc[idx].copy()
                    row['predicted_label'] = pred_label
                    row['probability'] = max_prob
                    filtered_rows.append(row)
        elif self.classifier_name == 'lstm_glove':
            for idx, (text, true_label) in enumerate(zip(texts, true_labels)):
                embedding = self.get_embedding(text)
                embedding = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
                    self.device)  # Add batch and sequence dimensions
                with torch.no_grad():
                    output = self.model(embedding)
                    probs = torch.nn.functional.softmax(output, dim=-1)
                    max_prob, pred_label = torch.max(probs, dim=1)
                    pred_label = self.label_encoder.inverse_transform([pred_label.item()])[0]
                    if pred_label == true_label and max_prob.item() >= threshold:
                        row = data.iloc[idx].copy()
                        row['predicted_label'] = pred_label
                        row['probability'] = max_prob.item()
                        filtered_rows.append(row)

        filtered_df = pd.DataFrame(filtered_rows)
        filtered_df.to_csv(output_path, index=False)
        print(f"Filtered dataset saved to {output_path}")


if __name__ == "__main__":
    classifier_name = "bert"
    threshold = 0.7
    generated_data = "generated_datasets/Llama3_8B_ATIS_100_augmented_data.csv"
    output = "filtered_datasets/Llama3_8B_ATIS_100_augmented_data.csv"
    glove_file = "classifiers/embeddings/glove.6B.100d.txt"
    model_path = "classifiers/models/bert_model"
    label_encoder_path = "classifiers/models/label_encoder.pkl"

    dataset_filter = DatasetFilter(classifier_name, model_path, glove_file, label_encoder_path)
    dataset_filter.filter_dataset(generated_data, threshold, output)
