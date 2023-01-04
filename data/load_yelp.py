import os
import pandas as pd
import numpy as np
from constant import DATA_DIR


MULTI_ATTRS = {
    'user_id': ['_BcWyKQL16ndpBdggh2kNA', 'Xw7ZjaGfr0WNVt6s_5KZfA', '0Igx-a1wAstiBDerGxXk2A', '-G7Zkl1wIWBBmD0KRy_sCw', 'ET8n-r7glWYqZhuR6GcdNw', 'bYENop4BuQepBjM1-BI3fA', '1HM81n6n4iPIFU5d2Lokhw', 'fr1Hz2acAb3OaL3l6DyKNg', 'wXdbkFZsfDR7utJvbWElyA', 'Um5bfs5DH6eizgjH3xZsvg'],
    'stars': [0, 1, 2, 3, 4, 5],
    'useful': [0, 1, 2, 3, 4, 5],
    'funny': [0, 1, 2, 3, 4, 5],
    'cool': [0, 1, 2, 3, 4, 5],
}

BINARY_ATTRS = {}

def text_to_onehot(text: pd.Series):
    from sklearn.feature_extraction.text import CountVectorizer
    text.to_numpy()
    vectorizer = CountVectorizer(max_features = 3000)
    onehot = vectorizer.fit_transform(text.to_numpy())
    return onehot.toarray()

def dataset_path(main_attr, infer_attr):
    data_name = f"yelp-author_{main_attr}_{infer_attr}.npz"
    return os.path.join(DATA_DIR, data_name)

def load_data(main_attr, infer_attr):
    path = dataset_path(main_attr, infer_attr)
    if os.path.exists(path):
        file = np.load(path)
        return file['x'], file['y'], file['prop']
    else:
        return None, None, None

def save_data(main_attr, infer_attr, x, y, prop):
    path = dataset_path(main_attr, infer_attr)
    np.savez(path, x=x, y=y, prop=prop)

def load_yelp_author_with_attrs(main_attr, infer_attr):
    x, y, prop = load_data(main_attr, infer_attr)
    if x is not None and y is not None and prop is not None:
        return x, y, prop

    r_dtypes = {"stars": np.int32, 
            "useful": np.int32, 
            "funny": np.int32,
            "cool": np.int32} 

    data_path = os.path.join(DATA_DIR, 'yelp_academic_dataset_review.json')
    df = pd.read_json(data_path, lines=True, dtype=r_dtypes, orient='records')
    df = df[['text', main_attr, infer_attr]]

    topn_infer_attrs = MULTI_ATTRS[infer_attr]
    filtered_df = df[df[infer_attr].isin(topn_infer_attrs)]


    x = text_to_onehot(filtered_df['text'])
    y = filtered_df[main_attr].to_numpy()
    prop = filtered_df[infer_attr].to_numpy()

    save_data(main_attr, infer_attr, x, y, prop)

    return x, y, prop


if __name__ == "__main__":
    load_yelp_author_with_attrs('review', 'stars')