import torch
import torch.nn as nn
from torch_scatter import scatter
import torch.nn.functional as F
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj,bipartite_subgraph, softmax)
from torch_geometric.nn.models import MLP
class Text_enc(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hidden_dim = params.text_dim
        self.u = nn.Linear(params.text_dim, 1)
        self.W = nn.Linear(params.text_dim , params.text_dim)

    def forward(self, ent_num, Textid, Text, Text_rel):
        # print(edge_index.device)

        a_v = self.W(Text)
        o = self.u(Text_rel)
        alpha = softmax(o, Textid, None, ent_num)
        text = scatter(alpha * a_v, index=Textid, dim=0, dim_size=ent_num, reduce='mean')

        return text


class FeatureMapping(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.in_dims = {'Stru': params.stru_dim, 'Text': params.text_dim, 'IMG': params.hidden_dim,
                        'Temporal': params.time_dim, 'Numerical': params.time_dim}
        self.out_dim = params.hidden_dim
        modals = ['Stru', 'Text', 'IMG', 'Temporal', 'Numerical']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':

            self.W_list = {
                modal: MLP(in_channels=self.in_dims[modal], out_channels=self.out_dim,
                           hidden_channels=params.MLP_hidden_dim, num_layers=params.MLP_num_layers,
                           dropout=params.MLP_dropout, norm=None).cuda() for modal in modals
            }
        else:
            self.W_list = {
                modal: MLP(in_channels=self.in_dims[modal], out_channels=self.out_dim,
                           hidden_channels=params.MLP_hidden_dim, num_layers=params.MLP_num_layers,
                           dropout=params.MLP_dropout, norm=None) for modal in modals
            }
        self.W_list = nn.ModuleDict(self.W_list)

    def forward(self, features):
        new_features = {}
        modals = ['Text']

        for modal, feature in features.items():
            if modal not in modals:
                continue
            # print(modal,feature.device)
            new_features[modal] = self.W_list[modal](feature)
        mean_feature = torch.mean(torch.stack(list(new_features.values())), dim=0)
        return new_features, mean_feature


class MMFeature(nn.Module):
    def __init__(self, n_ent, params):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.params = params
        self.n_ent = n_ent
        self.feature_mapping = FeatureMapping(params)
        self.text_model = Text_enc(params)
        self.in_dims = {'Stru': params.stru_dim, 'Text': params.text_dim, 'IMG': params.img_dim}
        self.out_dim = params.hidden_dim
        modals = ['Text', 'IMG']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.W_list = {
            modal: MLP(in_channels=self.in_dims[modal], out_channels=self.out_dim,
                       hidden_channels=params.MLP_hidden_dim, num_layers=params.MLP_num_layers,
                       dropout=params.MLP_dropout, norm=None).to(self.device) for modal in modals
        }
        self.W_list = nn.ModuleDict(self.W_list)

    def forward(self, img_features = None,att_features= None,att_rel_features= None, att_ids=None):
        features = {'IMG': self.W_list['IMG'](img_features),
                    'Text': self.W_list['Text'](self.text_model(self.n_ent, att_ids, att_features, att_rel_features))}

        mean_feature = torch.mean(torch.stack(list(features.values())), dim=0)
        return features, mean_feature


class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x:x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        # +3 for self-loop, alignment and alignment-inverse
        self.rela_embed = nn.Embedding(2 * n_rel + 3, in_dim)

        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, hidden, edges, n_node, old_nodes_new_idx):
        # edges:  [batch_idx, head, rela, tail, old_idx, new_idx]
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]

        hs = hidden[sub]
        hr = self.rela_embed(rel)

        # r_idx = edges[:, 0]
        # h_qr = self.rela_embed(q_rel)[r_idx]

        message = hs + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr))))
        message = alpha * message

        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))

        return hidden_new


class MASGNN(torch.nn.Module):
    def __init__(self, params, loader):
        super(MASGNN, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.mm = params.mm
        self.n_rel = loader.n_rel
        self.n_ent = loader.n_ent
        self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)

        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)  # get score
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)
        if self.mm:
            self.img_features = F.normalize(torch.FloatTensor(self.loader.images_list)).cuda()
            self.att_features = torch.FloatTensor(self.loader.att_features).cuda()
            self.att_rel_features = torch.FloatTensor(self.loader.att_rel_features).cuda()
            self.att_ids = torch.LongTensor(self.loader.att_ids).cuda()
            self.mmfeature = MMFeature(self.n_ent, params)


    def forward(self, subs, mode='train'):
        n = len(subs)
        q_sub = torch.LongTensor(subs).cuda()
        nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)



        if self.mm:
            features, mean_feature = self.mmfeature(img_features=self.img_features, att_features=self.att_features,
                                                    att_rel_features=self.att_rel_features, att_ids=self.att_ids)
            hidden = mean_feature[nodes[:, 1]]
            h0 = mean_feature[nodes[:, 1]].unsqueeze(0)
        else:
            h0 = torch.zeros((1, n, self.hidden_dim)).cuda()
            hidden = torch.zeros(n, self.hidden_dim).cuda()




        scores_all = []
        for i in range(self.n_layer):
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), mode=mode)
            # print(nodes)
            # print(edges)
            # print(old_nodes_new_idx)
            # print(hidden)
            # print(h0)
            hidden = self.gnn_layers[i](hidden, edges, nodes.size(0), old_nodes_new_idx)
            # print(hidden)

            if self.mm:
                h0 = mean_feature[nodes[:, 1]].unsqueeze(0).cuda().index_copy_(1, old_nodes_new_idx, h0)
            else:
                h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

        scores = self.W_final(hidden).squeeze(-1)
        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()  # non_visited entities have 0 scores
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores
        return scores_all



