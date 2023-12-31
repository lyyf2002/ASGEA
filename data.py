import torch
import random
import json
import numpy as np
import pdb
import torch.distributed as dist
import os
import os.path as osp
from collections import Counter
import pickle
import torch.nn.functional as F
from transformers import BertTokenizer
import torch.distributed
from tqdm import tqdm
import re

from utils import get_topk_indices, get_adjr


class EADataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class Collator_base(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, batch):
        # pdb.set_trace()

        return np.array(batch)


# def load_data(logger, args):
#     assert args.data_choice in ["DWY", "DBP15K", "FBYG15K", "FBDB15K"]
#     if args.data_choice in ["DWY", "DBP15K", "FBYG15K", "FBDB15K"]:
#         KGs, non_train, train_ill, test_ill, eval_ill, test_ill_ = load_eva_data(logger, args)
#
#     elif args.data_choice in ["FBYG15K_attr", "FBDB15K_attr"]:
#         pass
#
#     return KGs, non_train, train_ill, test_ill, eval_ill, test_ill_
#


def load_eva_data(args):
    if "OEA" in args.data_choice:
        file_dir = osp.join(args.data_path, "OpenEA", args.data_choice)
    else:
        file_dir = osp.join(args.data_path, args.data_choice, args.data_split)
    lang_list = [1, 2]
    ent2id_dict, ills, triples, r_hs, r_ts, ids = read_raw_data(file_dir, lang_list)
    e1 = os.path.join(file_dir, 'ent_ids_1')
    e2 = os.path.join(file_dir, 'ent_ids_2')
    left_ents,left_id2name = get_ids(e1,file_dir)
    right_ents,right_id2name = get_ids(e2,file_dir)
    id2name = {**left_id2name, **right_id2name}
    if not args.data_choice == "DBP15K" and not args.data_choice == "OpenEA":
        id2rel = get_id2rel(os.path.join(file_dir, 'id2relation.txt'))
    elif args.data_choice == "OpenEA":
        id2rel = get_id2rel(os.path.join(file_dir, 'rel_ids'))
    else:
        id2rel = None
    ENT_NUM = len(ent2id_dict)
    REL_NUM = len(r_hs)
    np.random.shuffle(ills)
    if args.data_choice == "OpenEA":
        img_vec_path = osp.join(args.data_path, f"OpenEA/pkl/{args.data_split}_id_img_feature_dict.pkl")
    elif "FB" in file_dir:
        img_vec_path = osp.join(args.data_path, f"pkls/{args.data_choice}_id_img_feature_dict.pkl")
    else:
        # fr_en
        split = file_dir.split("/")[-1]
        img_vec_path = osp.join(args.data_path, "pkls", args.data_split + "_GA_id_img_feature_dict.pkl")

    assert osp.exists(img_vec_path)
    img_features = load_img(ENT_NUM, img_vec_path)
    print(f"image feature shape:{img_features.shape}")

    if args.word_embedding == "glove":
        word2vec_path = os.path.join(args.data_path, "embedding", "glove.6B.300d.txt")
    elif args.word_embedding == 'bert':
        pass
    else:
        raise Exception("error word embedding")

    name_features = None
    char_features = None
    if args.data_choice == "DBP15K" and (args.w_name or args.w_char):

        assert osp.exists(word2vec_path)
        ent_vec, char_features = load_word_char_features(ENT_NUM, word2vec_path, args)
        name_features = F.normalize(torch.Tensor(ent_vec))
        char_features = F.normalize(torch.Tensor(char_features))
        print(f"name feature shape:{name_features.shape}")
        print(f"char feature shape:{char_features.shape}")

    if args.unsup:
        mode = args.unsup_mode
        if mode == "char":
            input_features = char_features
        elif mode == "name":
            input_features = name_features
        else:
            input_features = F.normalize(torch.Tensor(img_features))

        train_ill = visual_pivot_induction(args, left_ents, right_ents, input_features, ills)
    else:
        train_ill = np.array(ills[:int(len(ills) // 1 * args.data_rate)], dtype=np.int32)

    test_ill_ = ills[int(len(ills) // 1 * args.data_rate):]
    test_ill = np.array(test_ill_, dtype=np.int32)

    test_left = torch.LongTensor(test_ill[:, 0].squeeze())
    test_right = torch.LongTensor(test_ill[:, 1].squeeze())

    left_non_train = list(set(left_ents) - set(train_ill[:, 0].tolist()))

    right_non_train = list(set(right_ents) - set(train_ill[:, 1].tolist()))

    print(f"#left entity : {len(left_ents)}, #right entity: {len(right_ents)}")
    print(f"#left entity not in train set: {len(left_non_train)}, #right entity not in train set: {len(right_non_train)}")

    rel_features = load_relation(ENT_NUM, triples, 1000)
    print(f"relation feature shape:{rel_features.shape}")
    if 'OpenEA' in args.data_choice:
        a1 = os.path.join(file_dir, f'attr_triples_1')
        a2 = os.path.join(file_dir, f'attr_triples_2')
        att_features, num_att_left, num_att_right = load_attr_withNums(['oea', 'oea'], [a1, a2], ent2id_dict, file_dir,
                                                                       topk=args.topk)
    elif 'FB' in args.data_choice:
        a1 = os.path.join(file_dir, 'FB15K_NumericalTriples.txt')
        a2 = os.path.join(file_dir, 'DB15K_NumericalTriples.txt') if 'DB' in args.data_choice else os.path.join(file_dir, 'YAGO15K_NumericalTriples.txt')
        att_features, num_att_left, num_att_right = load_attr_withNums(['FB15K','DB15K'] if 'DB' in args.data_choice else ['FB15K','YAGO15K'],[a1, a2], ent2id_dict, file_dir, topk=0)
    else:
        att1,att2 = args.data_split.split('_')
        a1 = os.path.join(file_dir, f'{att1}_att_triples')
        a2 = os.path.join(file_dir, f'{att2}_att_triples')
        att_features, num_att_left, num_att_right = load_attr_withNums([att1,att2],[a1, a2], ent2id_dict, file_dir, topk=args.topk)
    print(f"attribute feature shape:{len(att_features)}")
    print("-----dataset summary-----")
    print(f"dataset:\t\t {file_dir}")
    print(f"triple num:\t {len(triples)}")
    print(f"entity num:\t {ENT_NUM}")
    print(f"relation num:\t {REL_NUM}")
    print(f"train ill num:\t {train_ill.shape[0]} \t test ill num:\t {test_ill.shape[0]}")
    print("-------------------------")

    eval_ill = None
    input_idx = torch.LongTensor(np.arange(ENT_NUM))

    # pdb.set_trace()
    # train_ill = EADataset(train_ill)
    # test_ill = EADataset(test_ill)

    return {
        'ent_num': ENT_NUM,
        'rel_num': REL_NUM,
        'images_list': img_features,
        'rel_features': rel_features,
        'att_features': att_features,
        'num_att_left': num_att_left,
        'num_att_right': num_att_right,
        'name_features': name_features,
        'char_features': char_features,
        'input_idx': input_idx,
        'triples': triples,
        'id2name':id2name,
        'id2rel':id2rel
    }, {"left": left_non_train, "right": right_non_train},left_ents,right_ents, train_ill, test_ill, eval_ill, test_ill_


def load_word2vec(path, dim=300):
    """
    glove or fasttext embedding
    """
    # print('\n', path)
    word2vec = dict()
    err_num = 0
    err_list = []

    with open(path, 'r', encoding='utf-8') as file:
        for line in tqdm(file.readlines(), desc="load word embedding"):
            line = line.strip('\n').split(' ')
            if len(line) != dim + 1:
                continue
            try:
                v = np.array(list(map(float, line[1:])), dtype=np.float64)
                word2vec[line[0].lower()] = v
            except:
                err_num += 1
                err_list.append(line[0])
                continue
    file.close()
    print("err list ", err_list)
    print("err num ", err_num)
    return word2vec


def load_char_bigram(path):
    """
    character bigrams of translated entity names
    """
    # load the translated entity names
    ent_names = json.load(open(path, "r"))
    # generate the bigram dictionary
    char2id = {}
    count = 0
    for _, name in ent_names:
        for word in name:
            word = word.lower()
            for idx in range(len(word) - 1):
                if word[idx:idx + 2] not in char2id:
                    char2id[word[idx:idx + 2]] = count
                    count += 1
    return ent_names, char2id


def load_word_char_features(node_size, word2vec_path, args):
    """
    node_size : ent num
    """
    name_path = os.path.join(args.data_path, "DBP15K", "translated_ent_name", "dbp_" + args.data_split + ".json")
    assert osp.exists(name_path)
    save_path_name = os.path.join(args.data_path, "embedding", f"dbp_{args.data_split}_name.pkl")
    save_path_char = os.path.join(args.data_path, "embedding", f"dbp_{args.data_split}_char.pkl")
    if osp.exists(save_path_name) and osp.exists(save_path_char):
        print(f"load entity name emb from {save_path_name} ... ")
        ent_vec = pickle.load(open(save_path_name, "rb"))
        print(f"load entity char emb from {save_path_char} ... ")
        char_vec = pickle.load(open(save_path_char, "rb"))
        return ent_vec, char_vec

    word_vecs = load_word2vec(word2vec_path)
    ent_names, char2id = load_char_bigram(name_path)

    # generate the word-level features and char-level features

    ent_vec = np.zeros((node_size, 300))
    char_vec = np.zeros((node_size, len(char2id)))
    for i, name in ent_names:
        k = 0
        for word in name:
            word = word.lower()
            if word in word_vecs:
                ent_vec[i] += word_vecs[word]
                k += 1
            for idx in range(len(word) - 1):
                char_vec[i, char2id[word[idx:idx + 2]]] += 1
        if k:
            ent_vec[i] /= k
        else:
            ent_vec[i] = np.random.random(300) - 0.5

        if np.sum(char_vec[i]) == 0:
            char_vec[i] = np.random.random(len(char2id)) - 0.5
        ent_vec[i] = ent_vec[i] / np.linalg.norm(ent_vec[i])
        char_vec[i] = char_vec[i] / np.linalg.norm(char_vec[i])

    with open(save_path_name, 'wb') as f:
        pickle.dump(ent_vec, f)
    with open(save_path_char, 'wb') as f:
        pickle.dump(char_vec, f)
    print("save entity emb done. ")
    return ent_vec, char_vec


def visual_pivot_induction(args, left_ents, right_ents, img_features, ills):

    l_img_f = img_features[left_ents]  # left images
    r_img_f = img_features[right_ents]  # right images

    img_sim = l_img_f.mm(r_img_f.t())
    topk = args.unsup_k
    two_d_indices = get_topk_indices(img_sim, topk * 100)
    del l_img_f, r_img_f, img_sim

    visual_links = []
    used_inds = []
    count = 0
    for ind in two_d_indices:
        if left_ents[ind[0]] in used_inds:
            continue
        if right_ents[ind[1]] in used_inds:
            continue
        used_inds.append(left_ents[ind[0]])
        used_inds.append(right_ents[ind[1]])
        visual_links.append((left_ents[ind[0]], right_ents[ind[1]]))
        count += 1
        if count == topk:
            break

    count = 0.0
    for link in visual_links:
        if link in ills:
            count = count + 1
    print(f"{(count / len(visual_links) * 100):.2f}% in true links")
    print(f"visual links length: {(len(visual_links))}")
    train_ill = np.array(visual_links, dtype=np.int32)
    return train_ill


def read_raw_data(file_dir, lang=[1, 2]):
    """
    Read DBP15k/DWY15k dataset.
    Parameters
    ----------
    file_dir: root of the dataset.
    Returns
    -------
    ent2id_dict : A dict mapping from entity name to ids
    ills: inter-lingual links (specified by ids)
    triples: a list of tuples (ent_id_1, relation_id, ent_id_2)
    r_hs: a dictionary containing mappings of relations to a list of entities that are head entities of the relation
    r_ts: a dictionary containing mappings of relations to a list of entities that are tail entities of the relation
    ids: all ids as a list
    """
    print('loading raw data...')

    def read_file(file_paths):
        tups = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    tups.append(tuple([int(x) for x in params]))
        return tups

    def read_dict(file_paths):
        ent2id_dict = {}
        ids = []
        for file_path in file_paths:
            id = set()
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    ent2id_dict[params[1]] = int(params[0])
                    id.add(int(params[0]))
            ids.append(id)
        return ent2id_dict, ids
    ent2id_dict, ids = read_dict([file_dir + "/ent_ids_" + str(i) for i in lang])
    ills = read_file([file_dir + "/ill_ent_ids"])
    triples = read_file([file_dir + "/triples_" + str(i) for i in lang])
    r_hs, r_ts = {}, {}
    for (h, r, t) in triples:
        if r not in r_hs:
            r_hs[r] = set()
        if r not in r_ts:
            r_ts[r] = set()
        r_hs[r].add(h)
        r_ts[r].add(t)
    assert len(r_hs) == len(r_ts)
    return ent2id_dict, ills, triples, r_hs, r_ts, ids


def loadfile(fn, num=1):
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def get_ids(fn,file_dir):
    ids = []
    id2name = {}
    fbid2name = {}
    if 'FB' in fn:
        with open(os.path.join(file_dir, 'fbid2name.txt'), encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                fbid2name[th[0]] = th[1]
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            ids.append(int(th[0]))
            name = th[1]
            if '<http://yago-knowledge.org/resource/' in name:
                name = name[1:-1].split('/')[-1]
            if 'FB' in fn:
                if name in fbid2name:
                    name = fbid2name[name]
            if '<http://dbpedia.org/resource/' in name:
                name = name[1:-1].split('/')[-1].replace('_', ' ')
            id2name[int(th[0])] = name

    return ids, id2name

def get_id2rel(fn):
    id2rel = {}
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            rel = th[1]
            if '/' in rel:
                rel = rel.split('/')[-1]
            id2rel[int(th[0])] = rel
    return id2rel

def get_ent2id(fns):
    ent2id = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                ent2id[th[1]] = int(th[0])
    return ent2id
def split_camel_case(input_string):
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', input_string)
    return words
def db_str(s):
    return ' '.join(split_camel_case(s[1:-1].split('/')[-1].replace('_',' ')))
def db_time(s):
    s = s.split("^^")[0][1:-1]
    if 'e' in s:
        return s
    
    if '-' not in s[1:]:
        return s
    s = s.split('-')
    y = int(s[0].replace('#','0'))
    m = int(s[1]) if s[1]!='##'else 1
    d = int(s[2]) if s[2]!='##' and s[2]!='' else 1
    return y + (m-1)/12 +(d-1)/30/12
def dbp_str(s):
    if '<'==s[0] and '>'==s[-1]:
        s = s[1:-1]
    t = s.split('/')[-1].replace('_',' ')
    t_ = ' '.join(split_camel_case(t))
    if t_ == '':
        return t
    return t_
    

def dbp_value(s):
    if '^^' in s:
        s = s.split("^^")[0]
        if ('<' == s[0] and '>' == s[-1]) or ('\"' == s[0] and '\"' == s[-1]):
            s = s[1:-1]
    elif '@' in s and s.index('@')>0:
        s = s.split('@')[0]
        if ('<' == s[0] and '>' == s[-1]) or ('\"' == s[0] and '\"' == s[-1]):
            s = s[1:-1]
        if s[-1]=='\"':
            s = s[:-1]
    else:
        if ('<' == s[0] and '>' == s[-1]) or ('\"' == s[0] and '\"' == s[-1]):
            s = s[1:-1]
        return s
    if 'e' in s:
        return s
    
    if '-' not in s[1:]:
        return s
    try:
        s_ = s.split('-')
        y = int(s_[0].replace('#','0'))
        m = int(s_[1]) if s_[1]!='##'else 1
        d = int(s_[2]) if s_[2]!='##' and s_[2]!='' else 1
        return y + (m-1)/12 +(d-1)/30/12
    except:
        return s



def load_attr_withNums(datas,fns, ent2id_dict, file_dir, topk=0):
    ans =  [load_attr_withNum(data,fn,ent2id_dict) for data,fn in zip(datas,fns)]
    if topk!=0:

        rels = []
        rels2index = {}
        rels2times = {}
        cur = 0
        att2rel = []
        for i, att in enumerate(ans[0]+ans[1]):
            if att[1] not in rels2index:
                rels2index[att[1]] = cur
                rels.append(att[1])
                cur += 1
                rels2times[att[1]] = 0
            rels2times[att[1]] += 1
            att2rel.append(rels2index[att[1]])
        att2rel = np.array(att2rel)

        rels_left = []
        rels2index_left = {}
        cur = 0
        att2rel_left = []
        for i, att in enumerate(ans[0]):
            if att[1] not in rels2index_left:
                rels2index_left[att[1]] = cur
                rels_left.append(att[1])
                cur += 1
            att2rel_left.append(rels2index_left[att[1]])
        att2rel_left = np.array(att2rel_left)


        rels_right = []
        rels2index_right = {}
        cur = 0
        att2rel_right = []
        for i, att in enumerate(ans[1]):
            if att[1] not in rels2index_right:
                rels2index_right[att[1]] = cur
                rels_right.append(att[1])
                cur += 1
            att2rel_right.append(rels2index_right[att[1]])
        att2rel_right = np.array(att2rel_right)

        rels_right = set(rels_right)
        rels_left = set(rels_left)
        rels_inter = rels_left.intersection(rels_right)
        if len(rels_inter)==0:
            rels_inter = rels
        # select topk
        rels_inter = sorted(rels_inter, key=lambda x: rels2times[x], reverse=True)[:topk]

        ans_ = []
        for i in ans[0]:
            if i[1] in rels_inter:
                ans_.append(i)
        num_left = len(ans_)
        for i in ans[1]:
            if i[1] in rels_inter:
                ans_.append(i)
        num_right = len(ans_)-num_left
        return ans_,num_left,num_right



        # num_att_left = len(rels2index)
        # att_rel_features = np.load(os.path.join(file_dir, 'att_rel_features.npy'), allow_pickle=True)
        # rels = torch.FloatTensor(att_rel_features).cuda()
        # sim_rels_left = torch.mm(rels[:num_att_left], rels[num_att_left:].T)
        # sim_rels_right = torch.mm(rels[num_att_left:], rels[:num_att_left].T)
        # # get the max sim at row
        # sim_rels_left = torch.max(sim_rels_left, dim=1)[0]
        # sim_rels_right = torch.max(sim_rels_right, dim=1)[0]
        # # get the topk rels
        # topk_rels_left = torch.topk(sim_rels_left, topk, dim=0)[1]
        # topk_rels_right = torch.topk(sim_rels_right, topk, dim=0)[1]
        #
        # topk_rels_left = topk_rels_left.cpu().numpy()
        # topk_rels_right = topk_rels_right.cpu().numpy()
        # # topk_rels = np.concatenate([topk_rels_left,topk_rels_right+num_att_left])
        #
        #
        # # contain topkrels
        # common_elements = np.in1d(att2rel, topk_rels_left)
        # common_elements_indices = list(np.where(common_elements)[0])
        # ans_ = []
        # for i in common_elements_indices:
        #     ans_.append(ans[0][i])
        # num_left = len(ans_)
        #
        # rels = []
        # rels2index = {}
        # cur = 0
        # att2rel = []
        # for i,att in enumerate(ans[1]):
        #     if att[1] not in rels2index:
        #         rels2index[att[1]] = cur
        #         rels.append(att[1])
        #         cur += 1
        #     att2rel.append(rels2index[att[1]])
        # att2rel = np.array(att2rel)
        # # contain topkrels
        # common_elements = np.in1d(att2rel, topk_rels_right)
        # common_elements_indices = list(np.where(common_elements)[0])
        # for i in common_elements_indices:
        #     ans_.append(ans[1][i])
        # num_right = len(ans_) - num_left
        # return ans_,num_left,num_right
            
        
        
        
        
    return ans[0]+ans[1], len(ans[0]), len(ans[1])
def load_attr_withNum(data, fn, ent2id):

    with open(fn, 'r',encoding='utf-8') as f:
        Numericals = f.readlines()
    if data == 'FB15K' or data == 'DB15K' or data=='YAGO15K':
        Numericals_ = list(set(Numericals))
        Numericals_.sort(key = Numericals.index)
        Numericals = Numericals_

    if data=='FB15K':
        Numericals = [i[:-1].split('\t') for i in Numericals]
        Numericals = [(ent2id[i[0]], i[1][1:-1].replace('http://rdf.freebase.com/ns/', '').split('.')[-1].replace('_',' '), i[2]) for i in
                      Numericals]
    elif data=='DB15K':
        Numericals = [i[:-1].split(' ') if '\t' not in i else i[:-1].split('\t') for i in Numericals]
        Numericals = [(ent2id[i[0]], db_str(i[1]), db_time(i[2])) for i in Numericals]

    elif data=='YAGO15K':
        Numericals = [i[:-1].split(' ') if '\t' not in i else i[:-1].split('\t') for i in Numericals]
        Numericals = [(ent2id[i[0]], db_str(i[1]), db_time(i[2])) for i in Numericals]
    elif data=='oea':
        Numericals = [i[:-1].split('\t') for i in Numericals]
        Numericals = [(ent2id[i[0]], dbp_str(i[1]), dbp_value(i[2])) for i in Numericals]
    else:
        Numericals = [i[:-1].split(' ') if '\t' not in i else i[:-1].split('\t') for i in Numericals]
        Numericals = [(ent2id[i[0][1:-1]], dbp_str(i[1]), dbp_value(' '.join(i[2:]))) for i in Numericals]
        
    return Numericals


# The most frequent attributes are selected to save space
def load_attr(fns, e, ent2id, topA=1000):
    cnt = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] not in ent2id:
                    continue
                for i in range(1, len(th)):
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    attr2id = {}
    # pdb.set_trace()
    topA = min(1000, len(fre))
    for i in range(topA):
        attr2id[fre[i][0]] = i
    attr = np.zeros((e, topA), dtype=np.float32)
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            attr[ent2id[th[0]]][attr2id[th[i]]] = 1.0
    return attr


def load_relation(e, KG, topR=1000):
    # (39654, 1000)
    rel_mat = np.zeros((e, topR), dtype=np.float32)
    rels = np.array(KG)[:, 1]
    top_rels = Counter(rels).most_common(topR)
    rel_index_dict = {r: i for i, (r, cnt) in enumerate(top_rels)}
    for tri in KG:
        h = tri[0]
        r = tri[1]
        o = tri[2]
        if r in rel_index_dict:
            rel_mat[h][rel_index_dict[r]] += 1.
            rel_mat[o][rel_index_dict[r]] += 1.
    return np.array(rel_mat)


def load_json_embd(path):
    embd_dict = {}
    with open(path) as f:
        for line in f:
            example = json.loads(line.strip())
            vec = np.array([float(e) for e in example['feature'].split()])
            embd_dict[int(example['guid'])] = vec
    return embd_dict


def load_img(e_num, path):
    img_dict = pickle.load(open(path, "rb"))
    # init unknown img vector with mean and std deviation of the known's
    imgs_np = np.array(list(img_dict.values()))
    mean = np.mean(imgs_np, axis=0)
    std = np.std(imgs_np, axis=0)
    # img_embd = np.array([np.zeros_like(img_dict[0]) for i in range(e_num)]) # no image
    # img_embd = np.array([img_dict[i] if i in img_dict else np.zeros_like(img_dict[0]) for i in range(e_num)])

    img_embd = np.array([img_dict[i] if i in img_dict else np.random.normal(mean, std, mean.shape[0]) for i in range(e_num)])
    print(f"{(100 * len(img_dict) / e_num):.2f}% entities have images")
    return img_embd
