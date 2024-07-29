import importlib
from utils import load_data
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
warnings.filterwarnings("ignore", category=FutureWarning)


class ModelTrainer:
    def __init__(self, classifiers, glove_file, num_labels):
        self.classifiers = classifiers
        self.glove_file = glove_file
        self.num_labels = num_labels

    def train_all(self, train_text, train_labels, test_text=None, test_labels=None):
        results = {}
        for clf_name, clf_module in self.classifiers.items():
            print(f"Training {clf_name}...")
            module_name, class_name = clf_module.rsplit('.', 1)
            module = importlib.import_module(module_name)
            clf_class = getattr(module, class_name)
            if clf_name == 'lstm_glove':
                clf_instance = clf_class(self.glove_file, self.num_labels)
            elif clf_name == 'svm_glove':
                clf_instance = clf_class(self.glove_file)
            elif clf_name == 'bert':
                clf_instance = clf_class(num_labels=self.num_labels)
            else:
                clf_instance = clf_class()

            clf_instance.train(train_text, train_labels)
            accuracy = clf_instance.evaluate(test_text, test_labels) if test_text is not None and test_labels is not None else None
            if accuracy is not None:
                print(f"{clf_name} Accuracy: {accuracy:.4f}")
                results[clf_name] = accuracy
            clf_instance.save(f'{clf_name}')
        return results


if __name__ == "__main__":
    train_text, y_train, test_text, y_test, le = load_data('datasets/ATIS/atis.train.csv', 'datasets/ATIS/atis.test.csv')
    num_labels = len(le.classes_)
    # need to download from here https://www.kaggle.com/datasets/sawarn69/glove6b100dtxt
    # and save in classifiers/embeddings
    glove_file = 'classifiers/embeddings/glove.6B.100d.txt'

    classifiers = {
        'svm_tfidf': 'classifiers.svm_tfidf.SVM_TFIDF',
        'svm_glove': 'classifiers.svm_glove.SVM_Glove',
        'lstm_glove': 'classifiers.lstm_glove.LSTM_Glove',
        'bert': 'classifiers.bert.BERT',
    }

    trainer = ModelTrainer(classifiers, glove_file, num_labels)
    results = trainer.train_all(train_text, y_train, test_text, y_test)
    print("Training Results:", results)
