import torch
import numpy as np
import time
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from models import MASGNN
from utils import cal_ranks, cal_performance

class BaseModel(object):
    def __init__(self, args, loader):
        self.model = MASGNN(args, loader)
        self.model.cuda()

        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_batch = args.n_batch
        self.n_rel = loader.n_rel
        self.left_ents = loader.left_ents
        self.right_ents = loader.right_ents
        self.shuffle = args.shuffle

        self.n_train = loader.n_train
        # self.n_valid = loader.n_valid
        self.n_test  = loader.n_test
        self.n_layer = args.n_layer

        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.t_time = 0

    def train_batch(self,):
        epoch_loss = 0
        i = 0

        batch_size = self.n_batch
        n_batch = self.n_train // batch_size + (self.n_train % batch_size > 0)
        if self.shuffle:
            self.loader.shuffle_train()

        t_time = time.time()
        self.model.train()
        for i in tqdm(range(n_batch)):
            start = i*batch_size
            end = min(self.n_train, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            triple = self.loader.get_batch(batch_idx)

            self.model.zero_grad()
            scores = self.model(triple[:,0])

            pos_scores = scores[[torch.arange(len(scores)).cuda(),torch.LongTensor(triple[:,2]).cuda()]]
            max_n = torch.max(scores, 1, keepdim=True)[0]
            loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1)))
            # gamma = 0.1
            # lambd = 1
            # tau = 1
            # max_n = torch.max(scores, 1, keepdim=True)[0]
            # scores = max_n - scores
            # pos_scores = scores[[torch.arange(len(scores)).cuda(), torch.LongTensor(triple[:, 2]).cuda()]]
            # # extend pos_scores to scores
            # pos_scores = pos_scores.unsqueeze(-1)
            # l = gamma + pos_scores - scores
            # ln = (l - l.mean(dim=-1, keepdim=True).detach()) / l.std(dim=-1, keepdim=True).detach()
            # # ln = (l - mu) / torch.sqrt(sig + 1e-6)
            # loss = torch.sum(torch.log(1 + torch.sum(torch.exp(lambd * ln + tau), 1)))

            loss.backward()
            self.optimizer.step()

            # avoid NaN
            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            epoch_loss += loss.item()
        self.scheduler.step()
        self.t_time += time.time() - t_time

        t_mrr,t_h1, t_h3, t_h5, t_h10, out_str = self.evaluate()
        return t_mrr,t_h1, t_h3, t_h5, t_h10, out_str

    def evaluate(self, ):
        batch_size = self.n_batch
        i_time = time.time()
        n_data = self.n_test
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        self.model.eval()
        for i in range(n_batch):
            start = i*batch_size
            end = min(n_data, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            triple = self.loader.get_batch(batch_idx, data='test')
            subs, rels, objs = triple[:,0],triple[:,1],triple[:,2]
            is_lefts = rels == self.n_rel*2+1
            scores = self.model(subs,'test').data.cpu().numpy()

            ranks = cal_ranks(scores, objs, is_lefts, len(self.left_ents))
            ranking += ranks
        ranking = np.array(ranking)
        t_mrr, t_h1, t_h3, t_h5, t_h10 = cal_performance(ranking)
        i_time = time.time() - i_time

        out_str = '[TEST] MRR:%.4f H@1:%.4f H@3:%.4f H@5:%.4f H@10:%.4f \t[TIME] inference:%.4f\n' % (t_mrr, t_h1, t_h3, t_h5, t_h10, i_time)
        return t_mrr,t_h1, t_h3, t_h5, t_h10, out_str

