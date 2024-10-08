import ktrain
from ktrain import text
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class DistilBERT:
    def __init__(self, model_name='distilbert-base-cased', maxlen=50, batch_size=10, learning_rate=5e-5, epochs=12,
                 early_stopping=4, reduce_on_plateau=2, model_save_path='distilbert_model'):
        self.model_name = model_name
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model_save_path = model_save_path
        self.early_stopping = early_stopping
        self.reduce_on_plateau = reduce_on_plateau
        self.transformer = None
        self.learner = None
        self.predictor = None

    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        labels = np.unique(y_train)
        # Initialize the Transformer model
        self.transformer = text.Transformer(self.model_name, maxlen=self.maxlen, class_names=labels)

        # Preprocess the training and validation data
        processed_train = self.transformer.preprocess_train(X_train, y_train)
        processed_test = self.transformer.preprocess_test(X_valid, y_valid) if X_valid is not None and y_valid is not None else None

        # Get the classifier model
        model = self.transformer.get_classifier()
        self.learner = ktrain.get_learner(model, train_data=processed_train, val_data=processed_test, batch_size=self.batch_size)

        # self.learner.lr_find(show_plot=True, max_epochs=5)

        # Train the model
        self.learner.autofit(self.learning_rate, self.epochs,
                             early_stopping=self.early_stopping, reduce_on_plateau=self.reduce_on_plateau)

        # Save the trained model and preprocessor
        self.predictor = ktrain.get_predictor(self.learner.model, preproc=self.transformer)

    def evaluate(self, X_test=None, y_test=None):
        if X_test is None or y_test is None:
            print("No test data provided. Skipping evaluation.")
            return None
        # Make predictions
        predictions = self.predictor.predict(X_test)

        # Calculate accuracy
        np_test_labels = np.array(y_test)
        np_predictions = np.array(predictions)
        result = (np_test_labels == np_predictions)
        accuracy = result.sum() / len(result)
        precision, recall, f1, _ = precision_recall_fscore_support(np_test_labels, np_predictions, average='weighted')

        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        return accuracy, precision, recall, f1

    def save(self, model_path):
        self.predictor.save(model_path)
        print(f"Model saved to {model_path}")