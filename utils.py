import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(train_path, test_path=None):
    train_df = pd.read_csv(train_path)
    train_text = train_df['text'].tolist()
    train_label = train_df['label']

    le = LabelEncoder()
    y_train = le.fit_transform(train_label)

    if test_path:
        test_df = pd.read_csv(test_path)
        test_text = test_df['text'].tolist()
        test_label = test_df['label']
        y_test = le.transform(test_label)
        return train_text, y_train, test_text, y_test, le

    return train_text, y_train, le