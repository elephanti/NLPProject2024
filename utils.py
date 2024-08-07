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