import os
import random

import torch
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict
from data import load_eva_data
import pickle
from tqdm import tqdm
import lmdb
class DataLoader:
    def __init__(self, args):

        KGs, non_train, left_ents, right_ents, train_ill, test_ill, eval_ill, test_ill_ = load_eva_data(args)
        ent_num = KGs['ent_num']
        rel_num = KGs['rel_num']
        self.images_list = KGs['images_list']
        self.rel_features = KGs['rel_features']
        self.att_features = KGs['att_features']
        self.att_ids = [i[0] for i in self.att_features]
        self.test_cache_url = os.path.join(args.data_path, args.data_choice, args.data_split, f'test_{args.data_rate}')
        self.test_cache = {}

        if args.mm:
            if os.path.exists(os.path.join(args.data_path, args.data_choice, args.data_split, 'att_features.npy')):
                self.att_features = np.load(os.path.join(args.data_path, args.data_choice, args.data_split, 'att_features.npy'), allow_pickle=True)
                self.att_rel_features = np.load(os.path.join(args.data_path, args.data_choice, args.data_split, 'att_rel_features.npy'), allow_pickle=True)
            else:
                self.att_features, self.att_rel_features = self.bert_feature()
                np.save(os.path.join(args.data_path, args.data_choice, args.data_split, 'att_features.npy'), self.att_features)
                np.save(os.path.join(args.data_path, args.data_choice, args.data_split, 'att_rel_features.npy'), self.att_rel_features)
        self.name_features = KGs['name_features']
        self.char_features = KGs['char_features']
        triples = KGs['triples']

        self.left_ents = left_ents
        self.right_ents = right_ents

        self.n_ent = ent_num
        self.n_rel = rel_num

        self.filters = defaultdict(lambda: set())

        self.fact_triple = triples

        self.train_triple = self.ill2triples(train_ill)
        self.valid_triple = eval_ill  # None
        self.test_triple = self.ill2triples(test_ill)

        # add inverse
        self.fact_data = self.double_triple(self.fact_triple)
        # self.train_data = np.array(self.double_triple(self.train_triple))
        # self.valid_data = self.double_triple(self.valid_triple)
        self.test_data = self.double_triple(self.test_triple, ill=True)
        self.test_data = np.array(self.test_data)
        self.train_data = self.double_triple(self.train_triple, ill=True)
        self.train_data = np.array(self.train_data)

        # self.KG,self.M_sub = self.load_graph(self.fact_data) # do it in shuffle_train
        self.tKG = self.load_graph(self.fact_data + self.double_triple(self.train_triple, ill=True))
        self.tKG = torch.LongTensor(self.tKG).cuda()

        # in torch
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent), 1), 2 * self.n_rel * np.ones((self.n_ent, 1)),
                              np.expand_dims(np.arange(self.n_ent), 1)], 1)
        self.fact_data = np.concatenate([np.array(self.fact_data), idd], 0)
        self.fact_data = torch.LongTensor(self.fact_data).cuda()
        # self.node2index = {}
        # for i, triple in enumerate(self.train_triple):
        #     h, r, t = triple
        #     assert h not in self.node2index
        #     assert t not in self.node2index
        #     self.node2index[h] = i
        #     self.node2index[t] = i
        # self.train_triple = torch.LongTensor(self.train_triple).cuda()


        self.n_test = len(self.test_data)
        self.n_train = len(self.train_data)
        self.shuffle_train()

        # if os.path.exists(self.test_cache_url):
        #     self.test_env = lmdb.open(self.test_cache_url)
        # else:
        #     self.test_env = lmdb.open(self.test_cache_url, map_size=200*1024 * 1024 * 1024, max_dbs=1)
        #     self.preprocess_test()

    def bert_feature(self, ):
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertModel.from_pretrained("bert-base-multilingual-cased").cuda()

        outputs = []
        texts = [a + ' is ' + str(v) for i,a,v in self.att_features]
        batch_size = 512
        sent_batch = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        for sent in sent_batch:

            encoded_input = tokenizer(sent, return_tensors='pt', padding=True, truncation=True, max_length=512)
            #cuda
            encoded_input.data['input_ids'] = encoded_input.data['input_ids'].cuda()
            encoded_input.data['attention_mask'] = encoded_input.data['attention_mask'].cuda()
            encoded_input.data['token_type_ids'] = encoded_input.data['token_type_ids'].cuda()
            with torch.no_grad():
                output = model(**encoded_input)
            outputs.append(output.pooler_output)
        outputs = torch.cat(outputs, dim=0)
        rels = [i[1] for i in self.att_features]
        batch_size = 512
        sent_batch = [rels[i:i + batch_size] for i in range(0, len(rels), batch_size)]
        rel_outputs = []
        for sent in sent_batch:
            encoded_input = tokenizer(sent, return_tensors='pt', padding=True, truncation=True, max_length=512)
            #cuda
            encoded_input.data['input_ids'] = encoded_input.data['input_ids'].cuda()
            encoded_input.data['attention_mask'] = encoded_input.data['attention_mask'].cuda()
            encoded_input.data['token_type_ids'] = encoded_input.data['token_type_ids'].cuda()
            with torch.no_grad():
                output = model(**encoded_input)
            rel_outputs.append(output.pooler_output)
        rel_outputs = torch.cat(rel_outputs, dim=0)
        del model
        return outputs.cpu().detach().numpy(), rel_outputs.cpu().detach().numpy()



    def ill2triples(self, ill):
        return [(i[0], self.n_rel * 2 + 1, i[1]) for i in ill]

    # def read_triples(self, filename):
    #     triples = []
    #     with open(os.path.join(self.task_dir, filename)) as f:
    #         for line in f:
    #             h, r, t = line.strip().split()
    #             h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
    #             triples.append([h, r, t])
    #             self.filters[(h, r)].add(t)
    #             self.filters[(t, r + self.n_rel)].add(h)
    #     return triples

    def double_triple(self, triples, ill=False):
        new_triples = []
        for triple in triples:
            h, r, t = triple
            new_triples.append([t, r + self.n_rel if not ill else r+1, h])
        return triples + new_triples

    def load_graph(self, triples):
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent), 1), 2 * self.n_rel * np.ones((self.n_ent, 1)),
                              np.expand_dims(np.arange(self.n_ent), 1)], 1)

        KG = np.concatenate([np.array(triples), idd], 0)
        # n_fact = len(KG)
        # M_sub = csr_matrix((np.ones((n_fact,)), (np.arange(n_fact), KG[:, 0])),
        #                    shape=(n_fact, self.n_ent))
        return KG


    def get_subgraphs(self, head_nodes, layer=3,mode='train'):
        all_edges = []
        for index,head_node in enumerate(head_nodes):
            all_edge = self.get_subgraph(head_node, index, layer, mode)
            all_edges.append(all_edge)
        all_nodes = []
        layer_edges = []
        old_nodes_new_idxs = []
        old_nodes = []
        for i in range(layer):
            edges = []
            for j in range(len(all_edges)):
                edges.append(all_edges[j][i])
            edges = torch.cat(edges, dim=0)
            edges = edges.long()

            head_nodes, head_index = torch.unique(edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
            tail_nodes, tail_index = torch.unique(edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)
            sampled_edges = torch.cat([edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)


            mask = sampled_edges[:, 2] == (self.n_rel * 2)
            old_node, old_idx = head_index[mask].sort()
            old_nodes_new_idx = tail_index[mask][old_idx]
            all_nodes.append(tail_nodes)
            layer_edges.append(sampled_edges)
            old_nodes_new_idxs.append(old_nodes_new_idx)
            old_nodes.append(old_node)


        return all_nodes, layer_edges, old_nodes_new_idxs, old_nodes
    #
    def get_subgraph(self, head_node, index, layer, mode, max_size=500):
        if mode == 'train':
        #     # set false to self.node2index[node]
        #     mask = torch.ones(len(self.train_triple), dtype=torch.bool).cuda()
        #     mask[self.node2index[head_node.item()]] = False
        #     support = self.train_triple[mask]
        #     reverse_support = support[:, [2, 1, 0]]
        #     reverse_support[:, 1] += 1
        #     support = torch.cat((support, reverse_support), dim=0)
        #     KG = torch.cat((support,self.fact_data),dim=0)
            KG=self.KG
            KG.long()
        else:
            KG = self.tKG
        row, col = KG[:, 0], KG[:, 2]
        node_mask = row.new_empty(self.n_ent, dtype=torch.bool)
        # edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
        subsets = [torch.LongTensor([head_node]).cuda()]
        raw_layer_edges = []
        for i in range(layer):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            edge_mask = torch.index_select(node_mask, 0, row)
            subsets.append(torch.unique(col[edge_mask]))
            raw_layer_edges.append(edge_mask)
            # nodes, edges, old_nodes_new_idx = self.get_neighbors(nodes.data.cpu().numpy())
        # delete target not in the other KG
        tail_node = self.left_ents if head_node.item() >= len(self.left_ents) else self.right_ents
        tail_node = torch.LongTensor(tail_node).cuda()
        node_mask_ = row.new_empty(self.n_ent, dtype=torch.bool)
        node_mask_.fill_(False)
        node_mask_[tail_node] = True
        tail_set = subsets[-1]
        node_mask.fill_(False)
        node_mask[tail_set] = True
        node_mask = node_mask & node_mask_
        layer_edges = []
        for i in reversed(range(layer)):
            edge_mask = torch.index_select(node_mask, 0, col)
            edge_mask = edge_mask & raw_layer_edges[i]
            node_mask_.fill_(False)
            node_mask_[row[edge_mask]] = True
            node_mask = node_mask | node_mask_
            layer_edges.append(KG[edge_mask])
        layer_edges = layer_edges[::-1]
        batched_edges = []
        for i in range(layer):
            layer_edges[i] = torch.unique(layer_edges[i], dim=0)
            batched_edges.append(torch.cat([torch.ones(len(layer_edges[i])).unsqueeze(1).cuda() * index, layer_edges[i]], 1))
        return batched_edges

    # def get_neighbors(self, nodes, mode='train', n_hop=0):
    #     if mode == 'train':
    #         KG = self.KG
    #         M_sub = self.M_sub
    #     else:
    #         KG = self.tKG
    #         M_sub = self.tM_sub
    #         # if self.test_cache
    #
    #     # nodes: n_node x 2 with (batch_idx, node_idx)
    #     node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])), shape=(self.n_ent, nodes.shape[0])) # (n_ent, batch_size)
    #     edge_1hot = M_sub.dot(node_1hot)
    #     edges = np.nonzero(edge_1hot)
    #     sampled_edges = np.concatenate([np.expand_dims(edges[1], 1), KG[edges[0]]],
    #                                    axis=1)  # (batch_idx, head, rela, tail)
    #     sampled_edges = torch.LongTensor(sampled_edges).cuda()
    #
    #     # index to nodes
    #     head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
    #     tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)
    #
    #     sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
    #
    #     mask = sampled_edges[:, 2] == (self.n_rel * 2)
    #     _, old_idx = head_index[mask].sort()
    #     old_nodes_new_idx = tail_index[mask][old_idx]
    #
    #     return tail_nodes, sampled_edges, old_nodes_new_idx

    # def get_neighbor(self, node, mode='train', n_hop=0):
    #     if mode == 'train':
    #         # set false to self.node2index[node]
    #         mask = torch.ones(len(self.train_triple), dtype=torch.bool)
    #         mask[self.node2index[node]] = False
    #         KG = torch.cat(self.train_triple[mask],self.fact_data)
    #
    #     else:
    #         KG = self.tKG
    #         # if self.test_cache
    #
    #     # nodes: n_node x 2 with (batch_idx, node_idx)
    #     # node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])), shape=(self.n_ent, nodes.shape[0])) # (n_ent, batch_size)
    #     # edge_1hot = M_sub.dot(node_1hot)
    #     edges = KG[:, 0]==node
    #     edges = np.nonzero(edges)
    #     sampled_edges =  KG[edges[0]]  # (head, rela, tail)
    #     sampled_edges = torch.LongTensor(sampled_edges).cuda()
    #
    #     # index to nodes
    #     head_nodes, head_index = torch.unique(sampled_edges[:, 1], dim=0, sorted=True, return_inverse=True)
    #     tail_nodes, tail_index = torch.unique(sampled_edges[:, 3], dim=0, sorted=True, return_inverse=True)
    #
    #     sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
    #
    #     # mask = sampled_edges[:, 2] == (self.n_rel * 2)
    #     # _, old_idx = head_index[mask].sort()
    #     # old_nodes_new_idx = tail_index[mask][old_idx]
    #
    #     return tail_nodes, sampled_edges

    def get_batch(self, batch_idx, steps=2, data='train'):
        if data == 'train':
            return self.train_data[batch_idx]
        if data == 'valid':
            return None
        if data == 'test':
            return self.test_data[batch_idx]

        # subs = []
        # rels = []
        # objs = []
        #
        # subs = query[batch_idx, 0]
        # rels = query[batch_idx, 1]
        # objs = np.zeros((len(batch_idx), self.n_ent))
        # for i in range(len(batch_idx)):
        #     objs[i][answer[batch_idx[i]]] = 1
        # return subs, rels, objs

    def shuffle_train(self, ):
        # fact_triple = np.array(self.fact_triple)
        # train_triple = np.array(self.train_triple)
        # all_triple = np.concatenate([fact_triple, train_triple], axis=0)
        # n_all = len(all_triple)
        # rand_idx = np.random.permutation(n_all)
        # all_triple = all_triple[rand_idx]
    
        # random shuffle train_triples
        random.shuffle(self.train_triple)
        # support/query split 3/1
        support_triple = self.train_triple[:len(self.train_triple) * 3 // 4]
        query_triple = self.train_triple[len(self.train_triple) * 3 // 4:]
        # add inverse triples
        support_triple = self.double_triple(support_triple, ill=True)
        query_triple = self.double_triple(query_triple, ill=True)
        support = torch.LongTensor(support_triple).cuda()
        self.KG = torch.cat((support,self.fact_data),dim=0)
        # now the fact triples are fact_triple + support_triple
        # self.KG, self.M_sub = self.load_graph(self.fact_data + support_triple)
        self.n_train = len(query_triple)
        self.train_data = np.array(query_triple)
    
        # # increase the ratio of fact_data, e.g., 3/4->4/5, can increase the performance
        # self.fact_data = self.double_triple(all_triple[:n_all * 3 // 4].tolist())
        # self.train_data = np.array(self.double_triple(all_triple[n_all * 3 // 4:].tolist()))
        # self.n_train = len(self.train_data)
        # self.KG,self.M_sub = self.load_graph(self.fact_data)
    
        print('n_train:', self.n_train, 'n_test:', self.n_test)

    def preprocess_test(self, ):
        batch_size = 4
        n_data = self.n_test
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        for i in tqdm(range(n_batch)):
            start = i * batch_size
            end = min(n_data, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            triple = self.get_batch(batch_idx, data='test')
            subs, rels, objs = triple[:, 0], triple[:, 1], triple[:, 2]
            print(subs, rels, objs)
            n = len(subs)
            q_sub = torch.LongTensor(subs).cuda()
            nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)
            for h in range(5):
                nodes, edges, old_nodes_new_idx = self.get_neighbors(nodes.data.cpu().numpy(), mode='test',
                                                                            n_hop=h)
                # to np
                # self.test_cache[(i, h)] = (nodes.cpu().numpy(), edges.cpu().numpy(), old_nodes_new_idx.cpu().numpy())
                # use lmdb write
                with self.test_env.begin(write=True) as txn:
                    txn.put(f'{i}_{h}'.encode(), pickle.dumps((nodes.cpu().numpy(), edges.cpu().numpy(), old_nodes_new_idx.cpu().numpy())))
        # pickle.dump(self.test_cache, open(self.test_cache_url, 'wb'))

    def get_test_cache(self, batch_idx, h):
        #use lmdb read
        with self.test_env.begin(write=False) as txn:
            nodes, edges, old_nodes_new_idx = pickle.loads(txn.get(f'{batch_idx}_{h}'.encode()))
        return nodes, edges, old_nodes_new_idx
        # return self.test_cache[(batch_idx, h)]


    # def save_cache(self):
    #     with open(self.cache_path, 'wb') as f:
    #         pickle.dump(self.edge_cache, f)
    #
    # def load_cache(self):
    #     with open(self.cache_path, 'rb') as f:
    #         self.edge_cache = pickle.load(f)
    #         print("load cache from {}".format(self.cache_path))
