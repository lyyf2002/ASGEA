import os
import argparse

import torch
import numpy as np
from load_data import DataLoader
from base_model import BaseModel
import time
from collections import OrderedDict

parser = argparse.ArgumentParser(description="Parser for MASEA")
parser.add_argument("--data_path", default="../data/mmkg", type=str, help="Experiment path")
parser.add_argument("--data_choice", default="FBYG15K", type=str, choices=["DBP15K", "DWY", "FBYG15K", "FBDB15K"],
                    help="Experiment path")
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
parser.add_argument('--n_layer', type=int, default=4)
parser.add_argument('--n_batch', type=int, default=10)
parser.add_argument("--lamda", type=float, default=0.5)

parser.add_argument("--MLP_hidden_dim", type=int, default=16)
parser.add_argument("--MLP_num_layers", type=int, default=2)
parser.add_argument("--MLP_dropout", type=float, default=0.2)

parser.add_argument("--n_ent", type=int, default=0)
parser.add_argument("--n_rel", type=int, default=0)

parser.add_argument("--stru_dim", type=int, default=16)
parser.add_argument("--text_dim", type=int, default=768)
parser.add_argument("--img_dim", type=int, default=4096)
parser.add_argument("--time_dim", type=int, default=32)
parser.add_argument("--out_dim", type=int, default=32)
parser.add_argument("--train_support", type=int, default=0)
parser.add_argument("--gnn_model", type=str, default='RS_GNN')
parser.add_argument("--mm", type=int, default=0)
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
parser.add_argument("--exp_name", default="EA_exp", type=str, help="Experiment name")
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
parser.add_argument("--data_split", default="norm", type=str, help="Experiment split",
                    choices=["dbp_wd_15k_V2", "dbp_wd_15k_V1", "zh_en", "ja_en", "fr_en", "norm"])
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


args = parser.parse_args()
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    args.perf_file = os.path.join(results_dir, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + '.txt')
    print(args)
    print(args, file=open(args.perf_file, 'a'))
    loader = DataLoader(args)
    model = BaseModel(args, loader)

    best_pr = 0
    best_t_roc = 0
    best_t_pr = 0

    best_str = ''
    wait_patient = 10
    epoch = 0

    best_mrr = 0

    while wait_patient > 0:
        epoch += 1
        mrr, out_str = model.train_batch()
        with open(args.perf_file, 'a+') as f:
            f.write(out_str)
        if mrr > best_mrr:
            best_mrr = mrr
            best_str = out_str
            print(str(epoch) + '\t' + best_str)
            with open(args.perf_file,'a+') as f:
                f.write("best at "+ str(epoch) + '\t' + best_str)
            wait_patient = 10
        else:
            wait_patient -= 1

    print(best_str)

