import os
import random

import torch
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict
from data import load_eva_data
import pickle

class DataLoader:
    def __init__(self, args):

        KGs, non_train, left_ents, right_ents, train_ill, test_ill, eval_ill, test_ill_ = load_eva_data(args)
        ent_num = KGs['ent_num']
        rel_num = KGs['rel_num']
        self.images_list = KGs['images_list']
        self.rel_features = KGs['rel_features']
        self.att_features = KGs['att_features']
        self.att_ids = [i[0] for i in self.att_features]
        self.test_cache_url = os.path.join(args.data_path, args.data_choice, args.data_split, f'test_{args.data_rate}.pkl')
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
        # self.KG,self.M_sub = self.load_graph(self.fact_data) # do it in shuffle_train
        self.tKG, self.tM_sub = self.load_graph(self.fact_data + self.double_triple(self.train_triple, ill=True))

        self.n_test = len(self.test_data)
        self.shuffle_train()

        if os.path.exists(self.test_cache_url):
            self.test_cache = pickle.load(open(self.test_cache_url, 'rb'))
        else:
            self.preprocess_test()

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
        n_fact = len(KG)
        M_sub = csr_matrix((np.ones((n_fact,)), (np.arange(n_fact), KG[:, 0])),
                           shape=(n_fact, self.n_ent))
        return KG, M_sub

    # def load_test_graph(self, triples):
    #     idd = np.concatenate([np.expand_dims(np.arange(self.n_ent), 1), 2 * self.n_rel * np.ones((self.n_ent, 1)),
    #                           np.expand_dims(np.arange(self.n_ent), 1)], 1)
    #
    #     self.tKG = np.concatenate([np.array(triples), idd], 0)
    #     self.tn_fact = len(self.tKG)
    #     self.tM_sub = csr_matrix((np.ones((self.tn_fact,)), (np.arange(self.tn_fact), self.tKG[:, 0])),
    #                              shape=(self.tn_fact, self.n_ent))

    # def load_query(self, triples):
    #     triples.sort(key=lambda x: (x[0], x[1]))
    #     trip_hr = defaultdict(lambda: list())
    #
    #     for trip in triples:
    #         h, r, t = trip
    #         trip_hr[(h, r)].append(t)
    #
    #     queries = []
    #     answers = []
    #     for key in trip_hr:
    #         queries.append(key)
    #         answers.append(np.array(trip_hr[key]))
    #     return queries, answers

    def get_neighbors(self, nodes, mode='train', n_hop=0):
        if mode == 'train':
            KG = self.KG
            M_sub = self.M_sub
        else:
            KG = self.tKG
            M_sub = self.tM_sub

        # nodes: n_node x 2 with (batch_idx, node_idx)
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:, 1], nodes[:, 0])), shape=(self.n_ent, nodes.shape[0])) # (n_ent, batch_size)
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        sampled_edges = np.concatenate([np.expand_dims(edges[1], 1), KG[edges[0]]],
                                       axis=1)  # (batch_idx, head, rela, tail)
        sampled_edges = torch.LongTensor(sampled_edges).cuda()

        # index to nodes
        head_nodes, head_index = torch.unique(sampled_edges[:, [0, 1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)

        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)

        mask = sampled_edges[:, 2] == (self.n_rel * 2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        return tail_nodes, sampled_edges, old_nodes_new_idx

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
        # now the fact triples are fact_triple + support_triple
        self.KG, self.M_sub = self.load_graph(self.fact_data + support_triple)
        self.n_train = len(query_triple)
        self.train_data = np.array(query_triple)

        # # increase the ratio of fact_data, e.g., 3/4->4/5, can increase the performance
        # self.fact_data = self.double_triple(all_triple[:n_all * 3 // 4].tolist())
        # self.train_data = np.array(self.double_triple(all_triple[n_all * 3 // 4:].tolist()))
        # self.n_train = len(self.train_data)
        # self.KG,self.M_sub = self.load_graph(self.fact_data)

        print('n_train:', self.n_train, 'n_test:', self.n_test)

    def preprocess_test(self, ):
        batch_size = self.n_batch
        n_data = self.n_test
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        for i in range(n_batch):
            start = i * batch_size
            end = min(n_data, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            triple = self.loader.get_batch(batch_idx, data='test')
            subs, rels, objs = triple[:, 0], triple[:, 1], triple[:, 2]
            is_lefts = rels == self.n_rel * 2 + 1
            n = len(subs)
            q_sub = torch.LongTensor(subs).cuda()
            nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(), q_sub.unsqueeze(1)], 1)
            for i in range(self.n_layer):
                nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), mode='test',
                                                                            n_hop=i)

    # def save_cache(self):
    #     with open(self.cache_path, 'wb') as f:
    #         pickle.dump(self.edge_cache, f)
    #
    # def load_cache(self):
    #     with open(self.cache_path, 'rb') as f:
    #         self.edge_cache = pickle.load(f)
    #         print("load cache from {}".format(self.cache_path))


if __name__ == '__main__':
