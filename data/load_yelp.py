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

def text_to_label(texts: pd.Series):
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer()
    vectorizer = vectorizer.fit(texts)
    tokenizer = vectorizer.build_tokenizer()

    n = 3000
    matrix = vectorizer.fit_transform(texts)
    freqs = zip(vectorizer.get_feature_names(), matrix.toarray().sum(axis=0))  
    # sort from largest to smallest
    topk_freqs = sorted(freqs, key=lambda x: -x[1])[:n]
    # build voacb dict
    vocab_dict = { word: index + 1 for index, (word, count) in enumerate(topk_freqs) }
    vocab_dict

    x = []
    max_len = 0

    for text in texts:
        ids = []
        for word in tokenizer(text):
            if word in vocab_dict:
                ids.append(vocab_dict[word])
        ids = np.array(ids, np.int32)
        max_len = max(len(ids), max_len)
        x.append(ids)

    for i, ids in enumerate(x):
        ids = np.concatenate((ids, np.zeros(max_len - len(ids), dtype = np.int32)))
        x[i] = ids

    x = np.array(x)
    return x

def dataset_path(main_attr, infer_attr):
    data_name = f"yelp-author_{main_attr}_{infer_attr}.npz"
    return os.path.join(DATA_DIR, data_name)

def load_data(main_attr, infer_attr):
    path = dataset_path(main_attr, infer_attr)
    if os.path.exists(path):
        file = np.load(path, allow_pickle=True)
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


    x = text_to_label(filtered_df['text'])
    y = filtered_df[main_attr].to_numpy()
    
    prop = filtered_df[infer_attr].to_numpy()
    if infer_attr == "user_id":
        prop = filtered_df[infer_attr].astype('category').cat.codes.to_numpy()
    if main_attr == "stars":
        y -= 1

    save_data(main_attr, infer_attr, x, y, prop)

    return x, y, prop


if __name__ == "__main__":
    load_yelp_author_with_attrs('stars', 'user_id')