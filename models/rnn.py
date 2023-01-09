import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchdp.per_sample_gradient_clip import PerSampleGradientClipper

from models import ModelFactory, Model

def nm_loss(pred, label):  # loss 정의
    loss = F.cross_entropy(pred, label)

    return torch.mean(loss)

@ModelFactory.register('yelp_author_gru')
class GRU(Model):  # 최종 classifier, optimizer 등
    def __init__(self, classes=5, input_shape=(5000), lr=0.05, n = 128, args=None):
        super().__init__()
        self.model = nn.Sequential(
            nn.Embedding(num_embeddings=3001, embedding_dim = 100),
            nn.GRU(input_size = 100, hidden_size = 128, batch_first=True),
        )
        self.fc = nn.Linear(128, classes)
        self.criterion = nm_loss
        self.optimizer = optim.SGD(self.parameters(), lr)

        if args.dp:
            self.clipper = PerSampleGradientClipper(self,args.clip)
            #self.criterion = nn.CrossEntropyLoss()#(reduction='none')

    def forward(self, x: torch.Tensor):
        x = x.long()
        hidden, _  = self.model(x)
        hidden_last = hidden[:, -1, :].squeeze()
        out = self.fc(hidden_last)
        out = F.softmax(out, dim=1)
        return out
