import json
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import random
import torch
import numpy as np
from load_data import DataLoader
from base_model import BaseModel
import time
from collections import OrderedDict
import networkx as nx
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Parser for MASEA")
parser.add_argument("--data_path", default="../data/mmkg", type=str, help="Experiment path")
parser.add_argument("--data_choice", default="FBYG15K", type=str, choices=["DBP15K", "DWY", "FBYG15K", "FBDB15K"],
                    help="Experiment path")
parser.add_argument("--data_split", default="norm", type=str, help="Experiment split",
                    choices=["dbp_wd_15k_V2", "dbp_wd_15k_V1", "zh_en", "ja_en", "fr_en", "norm"])
parser.add_argument("--data_rate", type=float, default=0.8, choices=[0.2, 0.3, 0.5, 0.8], help="training set rate")
parser.add_argument('--seed', type=str, default=1234)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--perf_file', type=str, default='perf.txt')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lamb', type=float, default=0.0002)
parser.add_argument('--decay_rate', type=float, default=0.991)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--attn_dim', type=int, default=5)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--act', type=str, default='relu')
parser.add_argument('--n_layer', type=int, default=5)
parser.add_argument('--n_batch', type=int, default=2)
parser.add_argument("--lamda", type=float, default=0.5)
parser.add_argument("--exp_name", default="EA_exp", type=str, help="Experiment name")
parser.add_argument("--MLP_hidden_dim", type=int, default=64)
parser.add_argument("--MLP_num_layers", type=int, default=3)
parser.add_argument("--MLP_dropout", type=float, default=0.2)

parser.add_argument("--n_ent", type=int, default=0)
parser.add_argument("--n_rel", type=int, default=0)

parser.add_argument("--stru_dim", type=int, default=16)
parser.add_argument("--text_dim", type=int, default=768)
parser.add_argument("--img_dim", type=int, default=2048)
parser.add_argument("--time_dim", type=int, default=32)
parser.add_argument("--out_dim", type=int, default=32)
parser.add_argument("--train_support", type=int, default=0)
parser.add_argument("--gnn_model", type=str, default='RS_GNN')
parser.add_argument("--mm", type=int, default=0)
parser.add_argument("--shuffle", type=int, default=1)
parser.add_argument("--meta", type=int, default=1)
parser.add_argument("--temperature", type=float, default=0.5)
parser.add_argument("--premm", type=int, default=0)
parser.add_argument("--withmm", type=int, default=1)
parser.add_argument("--update_step", type=int, default=20)
parser.add_argument("--update_step_test", type=int, default=20)
parser.add_argument("--update_lr", type=float, default=0.001)


# base

parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument("--save_model", default=0, type=int, choices=[0, 1])
parser.add_argument("--only_test", default=0, type=int, choices=[0, 1])

# torthlight
parser.add_argument("--no_tensorboard", default=False, action="store_true")

parser.add_argument("--dump_path", default="dump/", type=str, help="Experiment dump path")
parser.add_argument("--exp_id", default="001", type=str, help="Experiment ID")
parser.add_argument("--random_seed", default=42, type=int)


# --------- EA -----------

# parser.add_argument("--data_rate", type=float, default=0.3, help="training set rate")
#

# TODO: add some dynamic variable
parser.add_argument("--model_name", default="MEAformer", type=str, choices=["EVA", "MCLEA", "MSNEA", "MEAformer"],
                    help="model name")
parser.add_argument("--model_name_save", default="", type=str, help="model name for model load")

parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument("--scheduler", default="linear", type=str, choices=["linear", "cos", "fixed"])
parser.add_argument("--optim", default="adamw", type=str, choices=["adamw", "adam"])
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument("--adam_epsilon", default=1e-8, type=float)
parser.add_argument('--eval_epoch', default=100, type=int, help='evaluate each n epoch')
parser.add_argument("--enable_sota", action="store_true", default=False)

parser.add_argument('--margin', default=1, type=float, help='The fixed margin in loss function. ')
parser.add_argument('--emb_dim', default=1000, type=int, help='The embedding dimension in KGE model.')
parser.add_argument('--adv_temp', default=1.0, type=float,
                    help='The temperature of sampling in self-adversarial negative sampling.')
parser.add_argument("--contrastive_loss", default=0, type=int, choices=[0, 1])
parser.add_argument('--clip', type=float, default=1., help='gradient clipping')

# --------- EVA -----------

parser.add_argument("--hidden_units", type=str, default="128,128,128",
                    help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
parser.add_argument("--attn_dropout", type=float, default=0.0, help="dropout rate for gat layers")
parser.add_argument("--distance", type=int, default=2, help="L1 distance or L2 distance. ('1', '2')", choices=[1, 2])
parser.add_argument("--csls", action="store_true", default=False, help="use CSLS for inference")
parser.add_argument("--csls_k", type=int, default=10, help="top k for csls")
parser.add_argument("--il", action="store_true", default=False, help="Iterative learning?")
parser.add_argument("--semi_learn_step", type=int, default=10, help="If IL, what's the update step?")
parser.add_argument("--il_start", type=int, default=500, help="If Il, when to start?")
parser.add_argument("--unsup", action="store_true", default=False)
parser.add_argument("--unsup_k", type=int, default=1000, help="|visual seed|")

# --------- MCLEA -----------
parser.add_argument("--unsup_mode", type=str, default="img", help="unsup mode", choices=["img", "name", "char"])
parser.add_argument("--tau", type=float, default=0.1, help="the temperature factor of contrastive loss")
parser.add_argument("--alpha", type=float, default=0.2, help="the margin of InfoMaxNCE loss")
parser.add_argument("--with_weight", type=int, default=1, help="Whether to weight the fusion of different ")
parser.add_argument("--structure_encoder", type=str, default="gat", help="the encoder of structure view",
                    choices=["gat", "gcn"])
parser.add_argument("--ab_weight", type=float, default=0.5, help="the weight of NTXent Loss")

parser.add_argument("--projection", action="store_true", default=False, help="add projection for model")
parser.add_argument("--heads", type=str, default="2,2", help="heads in each gat layer, splitted with comma")
parser.add_argument("--instance_normalization", action="store_true", default=False,
                    help="enable instance normalization")
parser.add_argument("--attr_dim", type=int, default=100, help="the hidden size of attr and rel features")
parser.add_argument("--name_dim", type=int, default=100, help="the hidden size of name feature")
parser.add_argument("--char_dim", type=int, default=100, help="the hidden size of char feature")

parser.add_argument("--w_gcn", action="store_false", default=True, help="with gcn features")
parser.add_argument("--w_rel", action="store_false", default=True, help="with rel features")
parser.add_argument("--w_attr", action="store_false", default=True, help="with attr features")
parser.add_argument("--w_name", action="store_false", default=True, help="with name features")
parser.add_argument("--w_char", action="store_false", default=True, help="with char features")
parser.add_argument("--w_img", action="store_false", default=True, help="with img features")
parser.add_argument("--use_surface", type=int, default=0, help="whether to use the surface")

parser.add_argument("--inner_view_num", type=int, default=6, help="the number of inner view")
parser.add_argument("--word_embedding", type=str, default="glove", help="the type of word embedding, [glove|fasttext]",
                    choices=["glove", "bert"])
# projection head
parser.add_argument("--use_project_head", action="store_true", default=False, help="use projection head")
parser.add_argument("--zoom", type=float, default=0.1, help="narrow the range of losses")
parser.add_argument("--reduction", type=str, default="mean", help="[sum|mean]", choices=["sum", "mean"])

# --------- MEAformer -----------
parser.add_argument("--hidden_size", type=int, default=100, help="the hidden size of MEAformer")
parser.add_argument("--intermediate_size", type=int, default=400, help="the hidden size of MEAformer")
parser.add_argument("--num_attention_heads", type=int, default=5, help="the number of attention_heads of MEAformer")
parser.add_argument("--num_hidden_layers", type=int, default=2, help="the number of hidden_layers of MEAformer")
parser.add_argument("--position_embedding_type", default="absolute", type=str)
parser.add_argument("--use_intermediate", type=int, default=1, help="whether to use_intermediate")
parser.add_argument("--replay", type=int, default=0, help="whether to use replay strategy")
parser.add_argument("--neg_cross_kg", type=int, default=0,
                    help="whether to force the negative samples in the opposite KG")

# --------- MSNEA -----------
parser.add_argument("--dim", type=int, default=100, help="the hidden size of MSNEA")
parser.add_argument("--neg_triple_num", type=int, default=1, help="neg triple num")
parser.add_argument("--use_bert", type=int, default=0)
parser.add_argument("--use_attr_value", type=int, default=0)
# parser.add_argument("--learning_rate", type=int, default=0.001)
# parser.add_argument("--optimizer", type=str, default="Adam")
# parser.add_argument("--max_epoch", type=int, default=200)

# parser.add_argument("--save_path", type=str, default="save_pkl", help="save path")

# ------------ Para ------------
parser.add_argument('--rank', type=int, default=0, help='rank to dist')
parser.add_argument('--dist', type=int, default=0, help='whether to dist')
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--world-size', default=3, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
parser.add_argument("--local_rank", default=-1, type=int)

parser.add_argument("--nni", default=0, type=int)
args = parser.parse_args()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# use gpu 0
torch.cuda.set_device(args.gpu)


if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    args_str = f'{args.data_choice}_{args.data_split}_{args.data_rate}_lr{args.lr}_bs{args.n_batch}_hidden_dim{args.hidden_dim}_lamb{args.lamb}_dropout{args.dropout}_act{args.act}_decay_rate{args.decay_rate}'
    args.perf_file = os.path.join(results_dir, args.exp_name, args_str + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + '.txt')
    if not os.path.exists(os.path.join(results_dir, args.exp_name)):
        os.makedirs(os.path.join(results_dir, args.exp_name),exist_ok=True)
    if args.nni:
        import nni
        from nni.utils import merge_parameter
        nni_params = nni.get_next_parameter()
        args = merge_parameter(args, nni_params)
    print(args)
    print(args, file=open(args.perf_file, 'a'))
    loader = DataLoader(args)
    id2name = loader.id2name
    id2rel = loader.id2rel
    n_rel = loader.n_rel
    id2rel_reverse = {}
    for k, v in id2rel.items():
        id2rel_reverse[k+n_rel] = v+'_reverse'
    id2rel = {**id2rel , **id2rel_reverse}
    id2rel[2*n_rel] = 'self_loop'
    id2rel[2*n_rel+1] = 'anchor'
    id2rel[2*n_rel+2] = 'anchor_reverse'
    left_entity = len(loader.left_ents)
    
    batch_size = 1
    n_data = loader.n_test
    n_batch = n_data // batch_size + (n_data % batch_size > 0)

    for i in range(n_batch):
        start = i*batch_size
        end = min(n_data, (i+1)*batch_size)
        batch_idx = np.arange(start, end)
        triple = loader.get_batch(batch_idx, data='test')
        subs, rels, objs = triple[:,0],triple[:,1],triple[:,2]
        sub = subs[0]
        rel = rels[0]   
        obj = objs[0]
        edges = loader.get_vis_subgraph(sub, obj, 5)
        all_edges_size = sum([len(edge) for edge in edges])
        print(all_edges_size)
        if all_edges_size >100 or all_edges_size == 0:
            continue
        pos = {}
        x_pos = [-5,-3, -1, 1, 3, 5]
        g = {'nodes': [], 'edges': []}
        G = nx.DiGraph()
        for node in edges[0][:,0].unique():
            G.add_node(str(node.item()) + '_' + str(0), desc=id2name[node.item()] + '_' + str(0), layer=0)
            g['nodes'].append({'id': str(node.item()) + '_' + str(0), 'name': id2name[node.item()] + '_' + str(0),"class": 1 if node.item() < left_entity else 2 ,"imgsrc": "None","content": "None"} )
            pos[str(node.item()) + '_' + str(0)] = (x_pos[0], 0)
        for idx, edge in enumerate(edges):
            # node_1 = edge[:,0].unique()
            node_2 = edge[:,2].unique()
            size = len(node_2)

            for y, node in enumerate(node_2):
                G.add_node(str(node.item())+'_'+str(idx+1), desc=id2name[node.item()]+'_'+str(idx+1),layer=idx+1)
                g['nodes'].append({'id': str(node.item())+'_'+str(idx+1), 'name': id2name[node.item()]+'_'+str(idx+1),"class": 1 if node.item() < left_entity else 2,"imgsrc": "None","content": "None"} )
                pos[str(node.item())+'_'+str(idx+1)] = (x_pos[idx+1], 10/(size+1) * (y+1) - 5)
            for e in edge:
                g['edges'].append({'source': str(e[0].item())+'_'+str(idx), 'target': str(e[2].item())+'_'+str(idx+1), 'name': id2rel[e[1].item()]} )
                G.add_edge(str(e[0].item())+'_'+str(idx), str(e[2].item())+'_'+str(idx+1), name=id2rel[e[1].item()])


        # nodes = torch.cat([edges[:,0], edges[:,2]]).unique()
        # for node in nodes:
        #     G.add_node(node.item(), desc=id2name[node.item()])
        # for edge in edges:
        #     G.add_edge(edge[0].item(), edge[2].item(), name=id2rel[edge[1].item()])

        # draw graph with labels
        plt.figure(figsize=(16, 16), dpi=80)
        pos = nx.kamada_kawai_layout(G)
        # pos = nx.spring_layout(G)
        nx.draw(G, pos)
        nx.draw_networkx_nodes(G, pos=pos, nodelist=[str(sub.item()) + '_' + str(0),str(obj.item()) + '_' + str(5)], node_color='red', node_size=1000)
        node_labels = nx.get_node_attributes(G, 'desc')
        nx.draw_networkx_labels(G, pos, labels=node_labels)
        edge_labels = nx.get_edge_attributes(G, 'name')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.savefig(f'layer_{sub}_{rel}_{obj}.png', dpi=100)
        plt.close()
        json.dump(g, open(f'{sub}_{rel}_{obj}.json', 'w',encoding='utf-8'), indent=4)





