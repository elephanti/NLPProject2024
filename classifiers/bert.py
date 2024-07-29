import warnings
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning,
                        message="torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.")
warnings.filterwarnings("ignore", category=UserWarning,
                        message="User provided device_type of 'cuda', but CUDA is not available. Disabling")

# Set up logging to suppress specific messages
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BERT:
    def __init__(self, num_labels, max_len=128, batch_size=16, epochs=3, learning_rate=2e-5):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scaler = GradScaler(enabled=torch.cuda.is_available())

    def prepare_data(self, texts, labels):
        dataset = BERTDataset(texts, labels, self.tokenizer, self.max_len)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train_epoch(self, train_loader, optimizer, scheduler, loss_fn):
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            optimizer.zero_grad()
            with autocast(enabled=torch.cuda.is_available()):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()

            running_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions.double() / total_predictions
        return epoch_loss, epoch_accuracy

    def train(self, train_texts, train_labels):
        train_loader = self.prepare_data(train_texts, train_labels)
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, correct_bias=False)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

        for epoch in range(self.epochs):
            print(f"Starting epoch {epoch + 1}/{self.epochs}")
            epoch_loss, epoch_accuracy = self.train_epoch(train_loader, optimizer, scheduler, loss_fn)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    def evaluate(self, test_texts=None, test_labels=None):
        if test_texts is None or test_labels is None:
            print("No test data provided. Skipping evaluation.")
            return None

        test_loader = self.prepare_data(test_texts, test_labels)
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                _, preds = torch.max(logits, dim=1)

                correct_predictions += torch.sum(preds == labels)
                total_predictions += labels.size(0)

        accuracy = correct_predictions.double() / total_predictions
        return accuracy

    def save(self, model_path):
        self.model.save_pretrained(f'classifiers/models/{model_path}')
        self.tokenizer.save_pretrained(f'classifiers/models/{model_path}')
        print("Model and tokenizer saved.")