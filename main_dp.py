import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset_generic import Generic_WSI_Clinic_Dataset
from utils import get_split_loader, init_logger, set_seed
from utils.config import parse_config
from core_utils import train
from argparse import Namespace
import sys
from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
from dataset_modules.dataset_ood import WSI_OODDetection_Dataset

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score #Added to calculate F1-score

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=-1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--use_wandb', action='store_true', default=False,
                     help='use weight and biases for logging')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--split_file', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd','adamw'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'addmil'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str)
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='number of positive/negative patches to sample for clam')
parser.add_argument('--data_label_csv_path', type=str, default=None, help='data label directory') 
parser.add_argument('--datatype', type=str, default="pt", help='data type',choices=["npy","h5","pt"])   
parser.add_argument('--slide_col', type=str, default="slide_id")   
parser.add_argument('--label_col', type=str, default="label")    
parser.add_argument('--resume', action='store_true', default=False, help='resume training')
parser.add_argument('--hparamoptimisation_config',type=str,default=None)
parser.add_argument('--del_ckpt',action='store_true')
parser.add_argument('--wandbproject',type=str,default='toxpath')


def task_to_dataset(args):
    print("dataset args: ")
    print(args)
    if args.task == 'task_1_tumor_vs_normal':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(csv_path = args.data_label_csv_path,
                                data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                                patient_strat=False,
                                ignore=[])

    elif args.task == 'cscc_vs_noncscc':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(csv_path = args.data_label_csv_path,
                                data_dir= os.path.join(args.data_root_dir, 'features'),
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'non-cscc':0, 'cscc':1},
                                patient_strat=False,
                                ignore=[])

    elif args.task == 'task_2_tumor_subtyping':
        args.n_classes=3
        dataset = Generic_MIL_Dataset(csv_path = args.data_label_csv_path,
                                data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                                patient_strat= False,
                                ignore=[])

        if args.model_type in ['clam_sb', 'clam_mb']:
            assert args.subtyping 

    elif args.task == 'bcc_bin':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(csv_path = args.data_label_csv_path , 
                                data_dir= args.data_root_dir,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'normal': 0, 'bcc': 1},
                                slide_col= args.slide_col,
                                label_col = args.label_col,
                                case_id_col = "slide_stem",
                                patient_strat= False,
                                ignore=["unknown"],
                                datatype=args.datatype)

    elif args.task == 'mohs_bin':
        args.n_classes=2
        dataset = Generic_MIL_Dataset(csv_path = args.data_label_csv_path , 
                                data_dir= args.data_root_dir,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'normal': 0, 'bcc': 1},
                                patient_strat= False,
                                slide_col= args.slide_col,
                                label_col = args.label_col,
                                ignore=["unknown"],
                                datatype=args.datatype)

    elif args.task == 'abnormal_vs_normal':
        args.n_classes=2

        dataset = WSI_OODDetection_Dataset(
            csv_path=args.data_label_csv_path,
            shuffle=False,
            seed=args.seed,
            print_info=False,
            slide_col=args.slide_col,
            label_col=args.label_col,
            label_dict = {"0":0,"1":1},
            ignore=[],
            patient_strat=False,
            datatype=args.datatype,
            validate_inputs=False,
            data_dir=args.data_root_dir,
        )
    else:
        raise NotImplementedError
    
    return dataset




def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def run_training_ddp(rank, world_size, args):
    args = parser.parse_args()
    args.world_size = world_size

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    # different seed for each fold
    if args.seed == -1:
        args.seed = start

    # Logging & seed
    if rank == 0:
        init_logger(args)
    set_seed(args.seed + rank)

    # Data
    settings = {'num_splits': args.k, 
                'k_start': args.k_start,
                'k_end': args.k_end,
                'task': args.task,
                'max_epochs': args.max_epochs, 
                'results_dir': args.results_dir, 
                'lr': args.lr,
                'experiment': args.exp_code,
                'reg': args.reg,
                'label_frac': args.label_frac,
                'bag_loss': args.bag_loss,
                'seed': args.seed,
                'model_type': args.model_type,
                'model_size': args.model_size,
                "use_drop_out": args.drop_out,
                'weighted_sample': args.weighted_sample,
                'opt': args.opt,
                "hparamoptimisation_config": args.hparamoptimisation_config}

    if args.model_type in ['clam_sb', 'clam_mb']:
        settings.update({'bag_weight': args.bag_weight,
                            'inst_loss': args.inst_loss,
                            'B': args.B})


    dataset = task_to_dataset(args)
    args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))

    if args.split_dir is None:
        args.split_dir = os.path.join('splits', args.task+'_{}'.format(int(args.label_frac*100)))
    settings.update({'split_dir': args.split_dir})

    if rank == 0:
        if not os.path.isdir(args.results_dir):
            os.mkdir(args.results_dir)

        print('split_dir: ', args.split_dir)
        with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
            print(settings, file=f)
        f.close()

        print("################# Settings ###################")
        for key, val in settings.items():
            print("{}:  {}".format(key, val))       

        # create results directory if necessary
        if not os.path.isdir(args.results_dir):
            os.mkdir(args.results_dir)
            

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_val_f1 = [] #Added to calculate F1-score
    all_test_f1 = [] #Added to calculate F1-score
    folds = np.arange(start, end)

    for i in folds:
        if args.split_file is None:
            args.split_file = f"{args.split_dir}_{i}.csv"
        if rank == 0:
            print("split file: ", args.split_file)
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(csv_path=args.split_file,from_id=False)
        
        datasets = (train_dataset, val_dataset, test_dataset)
        results_val_dict, results_test_dict, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args,rank)
        
        results = {
            "val_results": results_val_dict,
            "test_results": results_test_dict,
            "val_auc": val_auc,
            "test_auc": test_auc,
            "val_acc": val_acc,
            "test_acc": test_acc,
        }


        gathered_results = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_results, results)

        if rank == 0:
            all_val_auc = []
            all_test_auc = []
            all_val_acc = []
            all_test_acc = []
            all_val_f1 = []
            all_test_f1 = []
            folds = list(range(world_size))

            for i, r in enumerate(gathered_results):
                df_val_class = pd.DataFrame.from_dict(r["val_results"], orient='index')
                df_val_class = df_val_class[["slide_id", "pred_class", "label"]]
                df_test_class = pd.DataFrame.from_dict(r["test_results"], orient='index')
                df_test_class = df_test_class[["slide_id", "pred_class", "label"]]

                all_val_auc.append(r["val_auc"])
                all_test_auc.append(r["test_auc"])
                all_val_acc.append(r["val_acc"])
                all_test_acc.append(r["test_acc"])
                all_val_f1.append(f1_score(df_val_class["label"], df_val_class["pred_class"], zero_division="warn"))
                all_test_f1.append(f1_score(df_test_class["label"], df_test_class["pred_class"], zero_division="warn"))

                filename = os.path.join(args.results_dir, f'split_{i}_results.pkl')
                save_pkl(filename, r["test_results"])

            final_df = pd.DataFrame({
                'folds': folds,
                'test_auc': all_test_auc,
                'val_auc': all_val_auc,
                'test_acc': all_test_acc,
                'val_acc': all_val_acc,
                'test_f1': all_test_f1,
                'val_f1': all_val_f1
            })

            if len(folds) != args.k:
                save_name = f'summary_partial_0_{len(folds)}.csv'
            else:
                save_name = 'summary.csv'

            final_df.to_csv(os.path.join(args.results_dir, save_name))

        dist.destroy_process_group()


def main():

    yaml_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    args = parse_config(yaml_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multi-GPU training.")

    world_size = torch.cuda.device_count()
    print(f"[INFO] Spawning {world_size} processes for DDP")
    mp.spawn(run_training_ddp, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
