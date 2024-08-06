import joblib
import torch
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import os
import nltk
import ktrain

nltk.download('punkt')


class LSTMGloveModel(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(LSTMGloveModel, self).__init__()
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.layer_norm = torch.nn.LayerNorm(hidden_dim * 2)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.permute(1, 0, 2).contiguous().view(h_n.size(1), -1)
        h_n = self.layer_norm(h_n)
        x = self.dropout(h_n)
        x = self.fc(x)
        return x


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

        if label_encoder_path:
            self.label_encoder = joblib.load(label_encoder_path)

        self.load_model()

    def load_model(self):
        if self.classifier_name == 'bert_ktrain':
            print(self.model_path)
            self.model = ktrain.load_predictor(self.model_path)
        elif self.classifier_name == 'svm_tfidf':
            self.model = joblib.load(f'{self.model_path}.pkl')
            self.vectorizer = joblib.load(f'{self.model_path}_vectorizer.pkl')
        elif self.classifier_name == 'svm_glove':
            self.model = joblib.load(f'{self.model_path}.pkl')
            self.glove_embeddings = self.load_glove_embeddings(self.glove_file)
        elif self.classifier_name == 'lstm_glove':
            self.model = LSTMGloveModel(embedding_dim=100, hidden_dim=64, output_dim=len(self.label_encoder.classes_))
            self.model.load_state_dict(torch.load(f'{self.model_path}.pth', map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.glove_embeddings = joblib.load(f'{self.model_path}_embeddings.pkl')

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
        embeddings = embeddings / max(1, len(tokens))
        return embeddings

    def filter_dataset(self, dataset_path, threshold, output_path):
        data = pd.read_csv(dataset_path)
        texts = data['text']
        true_labels = data['label']
        filtered_rows = []

        if self.classifier_name == 'bert_ktrain':
            for idx, (text, true_label) in enumerate(zip(texts, true_labels)):
                pred_prob = self.model.predict_proba([text])[0]
                pred_label = np.argmax(pred_prob)
                pred_label = self.label_encoder.inverse_transform([pred_label])[0]
                pred_prob = float(np.max(pred_prob))
                pred_prob_rounded = round(pred_prob, 4)
                # if pred_label == true_label and pred_prob >= threshold:
                row = data.iloc[idx].copy()
                row['predicted_label'] = pred_label
                row['probability'] = pred_prob_rounded
                filtered_rows.append(row)
                print(row)
        elif self.classifier_name == 'svm_tfidf':
            X = self.vectorizer.transform(texts)
            probs = self.model.predict_proba(X)
            preds = self.model.predict(X)
            for idx, (prob, pred, true_label) in enumerate(zip(probs, preds, true_labels)):
                max_prob = round(np.max(prob), 4)
                pred_label = self.label_encoder.inverse_transform([pred])[0]
                # if pred_label == true_label and max_prob >= threshold:
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
                max_prob = round(np.max(prob), 4)
                pred_label = self.label_encoder.inverse_transform([pred])[0]
                # if pred_label == true_label and max_prob >= threshold:
                row = data.iloc[idx].copy()
                row['predicted_label'] = pred_label
                row['probability'] = max_prob
                filtered_rows.append(row)
        elif self.classifier_name == 'lstm_glove':
            for idx, (text, true_label) in enumerate(zip(texts, true_labels)):
                embedding = self.get_embedding(text)
                embedding = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    output = self.model(embedding)
                    probs = torch.nn.functional.softmax(output, dim=-1)
                    max_prob, pred_label = torch.max(probs, dim=1)
                    max_prob = round(max_prob.item(), 4)
                    pred_label = self.label_encoder.inverse_transform([pred_label.item()])[0]
                    # if pred_label == true_label and max_prob >= threshold:
                    row = data.iloc[idx].copy()
                    row['predicted_label'] = pred_label
                    row['probability'] = max_prob
                    filtered_rows.append(row)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        filtered_df = pd.DataFrame(filtered_rows)
        filtered_df.to_csv(output_path, index=False)
        print(f"Filtered dataset saved to {output_path}")


if __name__ == "__main__":
    threshold = 0.7
    glove_file = "classifiers/embeddings/glove.6B.100d.txt"

    classifiers = ['svm_tfidf', 'svm_glove', 'lstm_glove', 'bert_ktrain']
    lambada = 'Lambada'
    gen_models = ['GPT2', 'Llama3', 'Mistral']

    for dataset_name in ["ATIS", "TREC"]:

        for classifier_name in classifiers:

            for gen_model in gen_models:

                if gen_model == 'Llama3':
                    size_model = "8B_"
                elif gen_model == 'Mistral':
                    size_model = "7B_"
                else:
                    size_model = ""

                os.makedirs(f"filtered_datasets/{lambada}/{gen_model}/{dataset_name}/{classifier_name}",
                            exist_ok=True)

                for num_samples in ["5", "10", "20", "50", "100"]:
                    model_path = os.path.join(f'classifiers/models/{dataset_name}',
                                              f'{dataset_name.lower()}_{num_samples}_subset', classifier_name)

                    generated_data = f"generated_datasets/{lambada}/{gen_model}/{dataset_name}/{gen_model}_{size_model}{dataset_name}_{num_samples}_augmented_data.csv"

                    output = f"filtered_datasets/{lambada}/{gen_model}/{dataset_name}/{classifier_name}/{gen_model}_{size_model}{dataset_name}_{num_samples}_augmented_data_{classifier_name}.csv"

                    label_encoder_path = os.path.join(f'classifiers/models/{dataset_name}',
                                                      f'{dataset_name.lower()}_{num_samples}_subset',
                                                      'label_encoder.pkl')

                    dataset_filter = DatasetFilter(classifier_name, model_path, glove_file, label_encoder_path)
                    dataset_filter.filter_dataset(generated_data, threshold, output)
