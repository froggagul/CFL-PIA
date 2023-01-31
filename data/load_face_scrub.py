import os
import pandas as pd
import numpy as np
from constant import DATA_DIR

SCRUB_DIR = os.path.join(DATA_DIR, 'face_scrub')

def get_face_scrub_attrs():
    MULTI_ATTRS = {
        'name': np.load(os.path.join(SCRUB_DIR, 'names.npy'), allow_pickle=True)
    }
    BINARY_ATTRS = {
        'gender': ['female', 'male']
    }
    return BINARY_ATTRS, MULTI_ATTRS 

def load_face_scrub_with_attrs(main_attr, infer_attr):
    assert main_attr == 'gender' and infer_attr == 'name', " only available for gender and identity"
    imgs = np.load(os.path.join(SCRUB_DIR, 'face_scrub_images.npy'))
    imgs = imgs.transpose(0, 3, 1, 2)
    df = pd.read_csv(os.path.join(SCRUB_DIR, 'facescrub.csv'))

    labels1 = df['gender']
    labels2 = df['name_index']
    x = imgs
    return x, labels1, labels2

if __name__ == "__main__":
    load_face_scrub_with_attrs('gender', 'name')
