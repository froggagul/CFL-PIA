import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

class AUROC():
    def __init__(self):
        self.y_true = [] # index 
        self.y_pred = [] # softmax value
        self.multi_class = None

    def _set_multi_class(self, shape):
        if self.multi_class is not None:
            return
        if shape[-1] > 2:
            self.multi_class = "ovr"
        else:
            self.multi_class = "raise"

    def update(self, true, pred):
        assert true.shape[0] == pred.shape[0], f"true({true.shape}) and pred({pred.shape}) shape is different"
        self._set_multi_class(pred.shape)
        pred = F.softmax(pred, dim = 1).numpy()
        
        self.y_true.append(true)
        self.y_pred.append(pred)

    def compute(self):
        y_true = np.concatenate(self.y_true, axis = 0)
        y_pred = np.concatenate(self.y_pred, axis = 0)
        
        assert self.multi_class is not None, "self.multi_class not set, you need to callupdate least once"
        if self.multi_class == "raise":
            y_pred = y_pred[:, 1]
        
        return roc_auc_score(y_true=y_true, y_score=y_pred, multi_class=self.multi_class)
