import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import joblib
from nltk.tokenize import word_tokenize
import nltk
from sklearn.metrics import precision_recall_fscore_support

nltk.download('punkt')


class LSTMGloveDataset(Dataset):
    def __init__(self, texts, labels, glove_embeddings, max_len=50, embedding_dim=100):
        self.texts = texts
        self.labels = labels
        self.glove_embeddings = glove_embeddings
        self.max_len = max_len
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = word_tokenize(text)
        embeddings = np.zeros((self.max_len, self.embedding_dim))
        for i, word in enumerate(tokens[:self.max_len]):
            embeddings[i] = self.glove_embeddings.get(word.lower(), np.zeros(self.embedding_dim))
        return {
            'embeddings': torch.tensor(embeddings, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }


class LSTMGloveModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(LSTMGloveModel, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n = h_n.permute(1, 0, 2).contiguous().view(h_n.size(1), -1)
        h_n = self.layer_norm(h_n)
        x = self.dropout(h_n)
        x = self.fc(x)
        return x


class LSTMGlove:
    def __init__(self, glove_file, num_labels, max_len=50, embedding_dim=100, hidden_dim=64, dropout_rate=0.5):
        self.glove_embeddings = self.load_glove_embeddings(glove_file)
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.model = LSTMGloveModel(embedding_dim, hidden_dim, num_labels, dropout_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def load_glove_embeddings(self, glove_file):
        embeddings_index = {}
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index

    def prepare_data(self, texts, labels):
        dataset = LSTMGloveDataset(texts, labels, self.glove_embeddings, self.max_len, self.embedding_dim)
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def train(self, X_train, y_train, epochs=20, lr=0.001, l2=0.0001):
        train_loader = self.prepare_data(X_train, y_train)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

        best_accuracy = 0
        best_model_state = None

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for batch in train_loader:
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(embeddings)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)
                total_predictions += labels.size(0)

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct_predictions.double() / total_predictions

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_state = self.model.state_dict()

            scheduler.step(epoch_loss)

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def evaluate(self, X_test=None, y_test=None):
        if X_test is None or y_test is None:
            print("No test data provided. Skipping evaluation.")
            return None

        test_loader = self.prepare_data(X_test, y_test)
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                embeddings = batch['embeddings'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(embeddings)
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels)
                total_predictions += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        accuracy = correct_predictions.double() / total_predictions
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return accuracy, precision, recall, f1

    def save(self, model_path):
        torch.save(self.model.state_dict(), f'{model_path}.pth')
        joblib.dump(self.glove_embeddings, f'{model_path}_embeddings.pkl')
        print("Model and embeddings saved.")
