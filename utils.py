import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_text = train_df['text']
    train_label = train_df['label']

    test_text = test_df['text']
    test_label = test_df['label']

    le = LabelEncoder()
    y_train = le.fit_transform(train_label)
    y_test = le.transform(test_label)

    joblib.dump(le, 'classifiers/models/label_encoder.pkl')

    return train_text, y_train, test_text, y_test, le
