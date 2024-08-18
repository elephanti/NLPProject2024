import glob
import os
import joblib
from classifiers.bert_ktrain import DistilBERT
from classifiers.lstm_glove import LSTMGlove
from classifiers.svm_glove import SVMGlove
from classifiers.svm_tfidf import SVMTFIDF
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# os.environ['TF_USE_LEGACY_KERAS'] = '1'
# print(os.environ['TF_USE_LEGACY_KERAS'])
# pip install ktrain

def load_data(train_path, full_path, test_path=None):
    train_df = pd.read_csv(train_path)
    train_text = train_df['text'].tolist()
    train_label = train_df['label']

    le = LabelEncoder()

    full_df = pd.read_csv(full_path)
    le.fit_transform(full_df['label'])

    y_train = le.transform(train_label)

    if test_path:
        test_df = pd.read_csv(test_path)
        test_text = test_df['text'].tolist()
        test_label = test_df['label']
        y_test = le.transform(test_label)
        return train_text, y_train, test_text, y_test, le

    return train_text, y_train, le

class ModelTrainer:
    def __init__(self, classifier_name, glove_file, num_labels):
        self.classifier = None
        self.classifier_name = classifier_name
        self.glove_file = glove_file
        self.num_labels = num_labels

    def train_classifier(self, X_train, y_train, model_path, X_test=None, y_test=None):
        print(f"Training {self.classifier_name}...")
        if self.classifier_name == 'lstm_glove':
            self.classifier = LSTMGlove
            clf_instance = self.classifier(self.glove_file, self.num_labels)
        elif self.classifier_name == 'svm_glove':
            self.classifier = SVMGlove
            clf_instance = self.classifier(self.glove_file)
        elif self.classifier_name == 'bert':
            self.classifier = DistilBERT
            clf_instance = self.classifier()
        else:
            self.classifier = SVMTFIDF
            clf_instance = self.classifier()

        clf_instance.train(X_train, y_train)

        accuracy = clf_instance.evaluate(X_test, y_test)
        if self.classifier_name in ['lstm_glove', 'bert_ktrain']:
            results = {"model": self.classifier_name, "test_accuracy": round(accuracy.item(), 4)}
        else:
            results = {"model": self.classifier_name, "test_accuracy": round(accuracy, 4)}

        clf_instance.save(f'{model_path}/{self.classifier_name}')

        return results


if __name__ == "__main__":
    # need to download from here https://www.kaggle.com/datasets/sawarn69/glove6b100dtxt
    # and save in classifiers/embeddings
    df_result = pd.DataFrame(columns=['dataset', 'model', 'test_accuracy'])
    glove_file = 'classifiers/embeddings/glove.6B.100d.txt'

    classifiers = {
        'svm_tfidf': SVMTFIDF,
        'svm_glove': SVMGlove,
        'lstm_glove': LSTMGlove,
        'bert_ktrain': DistilBERT
    }

    lambadas = ['Lambada', 'Lambada+', 'Lambada+Instruct']
    llms = ['Llama3', 'Mistral']
    dataset_names = ['ATIS', 'TREC']

    for name in dataset_names:
        for classifier in classifiers:
            # full training
            X_train, y_train, X_test, y_test, le = load_data(f'datasets/{name}/{name.lower()}.train.csv',
                                                             f'datasets/{name}/{name.lower()}.train.csv',
                                                             test_path=f'datasets/{name}/{name.lower()}.test.csv')
            model_path = f'final_results/models/{name.lower()}_train'
            os.makedirs(model_path, exist_ok=True)

            joblib.dump(le, f'{model_path}/label_encoder.pkl')

            trainer = ModelTrainer(classifier, glove_file, len(le.classes_))
            train_results = trainer.train_classifier(X_train, y_train, model_path, X_test=X_test, y_test=y_test)

            train_results["dataset"] = f'{name.lower()}_full'
            df_result = pd.concat([df_result, pd.DataFrame([train_results])], ignore_index=True)

        # augmented data
        for lambada in lambadas:
            for llm in llms:
                if lambada == 'Lambada+Instruct' and llm == 'Mistral':
                    break

                for classifier in classifiers:
                    directory_path = f'filtered_datasets/{lambada}/{llm}/{name}/{classifier}'

                    data_files = glob.glob(os.path.join(directory_path, '*.csv'))

                    for file in data_files:
                        dataset_file_name = file.replace(f'{directory_path}', "").replace('.csv', "").replace('/', "")
                        model_path = f'final_results/models/{dataset_file_name}'
                        os.makedirs(model_path, exist_ok=True)
                        X_train, y_train, X_test, y_test, le = load_data(file,
                                                                         f'datasets/{name}/{name.lower()}.train.csv',
                                                                         test_path=f'datasets/{name}/{name.lower()}.test.csv')
                        joblib.dump(le, f'{model_path}/label_encoder.pkl')

                        trainer = ModelTrainer(classifier, glove_file, len(le.classes_))
                        train_results = trainer.train_classifier(X_train, y_train, model_path, X_test=X_test, y_test=y_test)

                        train_results["dataset"] = dataset_file_name
                        df_result = pd.concat([df_result, pd.DataFrame([train_results])], ignore_index=True)

    df_result.to_csv("final_results/test_results.csv")
