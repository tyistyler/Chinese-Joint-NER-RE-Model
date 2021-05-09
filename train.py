import argparse
import torch
from models.casrel import Casrel
import numpy as np
import random
from config import config
from framework import framework

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Casrel', help='name of the model')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--multi_gpu', type=bool, default=False)
parser.add_argument('--dataset', type=str, default='NYT')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_epoch', type=int, default=50)
parser.add_argument('--test_epoch', type=int, default=1)
parser.add_argument('--train_triples', type=str, default='train_triples')
parser.add_argument('--dev_triples', type=str, default='dev_triples')
parser.add_argument('--test_triples', type=str, default='test_triples')
parser.add_argument('--max_seq_len', type=int, default=150)
parser.add_argument('--rel_num', type=int, default=24)
parser.add_argument('--period', type=int, default=50)

args = parser.parse_args()

set_seed(args.seed)

con = config.Config(args)

fw = framework.Framework(con)

model = {
    'Casrel': Casrel
}

fw.train(model[args.model_name])

