from .load_lfw import load_lfw_with_attrs
from .load_yelp import load_yelp_author_with_attrs
from .load_cifar100 import load_cifar100_with_attrs, load_cifar10_with_attrs
from .load_face_scrub import load_face_scrub_with_attrs, get_face_scrub_attrs

def load_data(data_type, main_attr, infer_attr):
    if data_type == "lfw":
        return load_lfw_with_attrs(main_attr, infer_attr)
    elif data_type == "yelp-author":
        return load_yelp_author_with_attrs(main_attr, infer_attr)
    elif data_type == "cifar100":
        return load_cifar100_with_attrs()
    elif data_type == "cifar10":
        return load_cifar10_with_attrs()
    elif data_type == "face_scrub":
        return load_face_scrub_with_attrs(main_attr, infer_attr)
    else:
        raise NotImplemented(f"{data_type} not exists in current version of code")

def load_attr(data_type):
    if data_type == "lfw":
        from data import LFW_BINARY_ATTRS, LFW_MULTI_ATTRS
        return LFW_BINARY_ATTRS, LFW_MULTI_ATTRS
    elif data_type == "yelp-author":
        from data import YELP_BINARY_ATTRS, YELP_MULTI_ATTRS
        return YELP_BINARY_ATTRS, YELP_MULTI_ATTRS
    elif data_type == "face_scrub":
        return get_face_scrub_attrs()
    else:
        raise NotImplemented(f"{data_type} not exists in current version of code")
