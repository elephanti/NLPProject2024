import glob
import os
import joblib
from classifiers.bert_ktrain import DistilBERT
from classifiers.lstm_glove import LSTMGlove
from classifiers.svm_glove import SVMGlove
from classifiers.svm_tfidf import SVMTFIDF
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
    def __init__(self, classifiers, glove_file, num_labels):
        self.classifiers = classifiers
        self.glove_file = glove_file
        self.num_labels = num_labels

    def train_all(self, X_train, y_train, model_path, X_test=None, y_test=None):
        results = []
        for clf_name, clf_module in self.classifiers.items():
            print(f"Training {clf_name}...")
            if clf_name == 'lstm_glove':
                clf_instance = clf_module(self.glove_file, self.num_labels)
            elif clf_name == 'svm_glove':
                clf_instance = clf_module(self.glove_file)
            elif clf_name == 'bert_ktrain':
                clf_instance = clf_module()
            else:
                clf_instance = clf_module()

            clf_instance.train(X_train, y_train)
            accuracy = clf_instance.evaluate(X_test, y_test) if X_test is not None and y_test is not None else None
            if accuracy is not None:
                if clf_name in ['lstm_glove', 'bert_ktrain']:
                    print(f"{clf_name} Accuracy: {accuracy.item():.4f}")
                    results.append({"model": clf_name, "val_accuracy": round(accuracy.item(), 4)})
                else:
                    print(f"{clf_name} Accuracy: {accuracy:.4f}")
                    results.append({"model": clf_name, "val_accuracy": round(accuracy, 4)})
            clf_instance.save(f'{model_path}/{clf_name}')

        return results


if __name__ == "__main__":
    # need to download from here https://www.kaggle.com/datasets/sawarn69/glove6b100dtxt
    # and save in classifiers/embeddings
    df_result = pd.DataFrame(columns=['data', 'model', 'val_accuracy'])
    glove_file = 'embeddings/glove.6B.100d.txt'

    classifiers = {
        'svm_tfidf': SVMTFIDF,
        'svm_glove': SVMGlove,
        'lstm_glove': LSTMGlove,
        'bert_ktrain': DistilBERT
    }

    dataset_names = ['ATIS', 'TREC']

    for name in dataset_names:
        print(name)
        directory_path = f'datasets/{name}/sampled_subsets/'

        data_files = glob.glob(os.path.join(directory_path, 'ver1', '*.csv'))
        for file in data_files:
            print(file)
            dataset_file_name = file.replace(f'{directory_path}', "").replace(f'ver1', "").replace('.csv', "").replace('/', "")
            model_path = f'classifiers/models/{name}/{dataset_file_name}'
            os.makedirs(model_path)

            X_train, y_train, X_test, y_test, le = load_data(file, test_path=f'datasets/{name}/{name.lower()}.valid.csv')
            joblib.dump(le, f'{model_path}/label_encoder.pkl')

            trainer = ModelTrainer(classifiers, glove_file, len(le.classes_))

            train_results = trainer.train_all(X_train, y_train, model_path, X_test=X_test, y_test=y_test)

            for i in range(len(classifiers)):

                train_results[i]["data"] = dataset_file_name
                df_result = pd.concat([df_result, pd.DataFrame([train_results[i]])], ignore_index=True)

    df_result.to_csv("classifiers/training_results.csv")
