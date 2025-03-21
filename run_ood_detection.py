import argparse
import pdb
import os
import math
from typing import Callable, List, Optional, TypeVar, Any, Dict, KeysView, Tuple, Union
import logging
log = logging.getLogger(__name__)
from pathlib import Path

# internal imports
from utils.utils import *
from utils.core_utils import get_pretrained_model
from dataset_modules.dataset_ood import Generic_WSI_OODDetection_Dataset

# pytorch imports
import torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


import pandas as pd
import numpy as np
from tqdm import tqdm

# ood detection imports

from pytorch_ood.api import RequiresFittingException
from pytorch_ood.utils import OODMetrics

from utils.ood_detection import precompute_logits, prepare_ood_detectors, RAW_FEATURE_SPACE, TRANSFORMED_FEATURE_SPACE, ATTENDED_FEATURE_SPACE, precompute_feats
from sklearn.metrics import f1_score #Added to calculate F1-score
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
***REMOVED***

def detectors_on_data(detectors,_dataloader,device,task_name,metric_calculators,results,model_based=False):
    for detector_name, detector in detectors.items():
        detector.fit(_dataloader,device=device)

    with torch.no_grad():
        if model_based:
            for detector_name,detector in detectors.items():
                print(detector_name)
                for x, y in tqdm(_dataloader,total=len(_dataloader)):
                    outlier_scores = detector.predict(x).view_as(y)
                    metric_calculators[detector_name].update(outlier_scores.to(device), y.to(device)) # add results to collection

                task_metrics = metric_calculators[detector_name].compute() # calculate metrics based on results
                metric_calculators[detector_name].reset()
                task_results =  {"Detector": detector_name, "Task": task_name} | task_metrics # concatenate dics
                results.append(task_results)

        else:
            for detector_name,detector in detectors.items():
                print(detector_name)
                for x, y in tqdm(_dataloader,total=len(_dataloader)):
                    outlier_scores = detector.predict_features(x).view_as(y)
                    metric_calculators[detector_name].update(outlier_scores.to(device), y.to(device)) # add results to collection

                task_metrics = metric_calculators[detector_name].compute() # calculate metrics based on results
                metric_calculators[detector_name].reset()
                task_results =  {"Detector": detector_name, "Task": task_name} | task_metrics # concatenate dics
                results.append(task_results)

    return metric_calculators,results


def run(args):
    seed_torch(args.seed)
    print(f"INFO: Evaluating classifier on {args.tasks} datasets.")

    """ Prepare the required components """
    model = get_pretrained_model(args,fold=0,for_ood=True)
    # model = make_branch_return_logits(model)

    test_dataloaders  = {}
    train_dataloaders = {}
    for task_name, task_test_dataset in zip(args.tasks,test_datasets):
        test_dataloaders[task_name] = get_split_loader(task_test_dataset, testing = args.testing, training=False)

    for id_data_name,id_data_dataset in train_datasets.items():
        train_dataloaders[id_data_name] = get_split_loader(id_data_dataset,testing=args.testing, training=False)

    logit_detectors, feature_detectors = prepare_ood_detectors(model,args)
    detectors = (logit_detectors | feature_detectors)
    detector_names = list( detectors.keys())

    metric_calculators = {k: OODMetrics() for k in detector_names}
    precomputed_logits = {}
    USE_LOGITS = False
    USE_FEATS = False
    if len(logit_detectors.keys()) > 0:
        USE_LOGITS = True
        for task_name, task_loader in test_dataloaders.items():
            precomputed_logits[task_name] = precompute_logits(model,task_loader,task_name, device,args,recompute=True)
    if len(feature_detectors.keys()) > 0:
        USE_FEATS = True

    """ Run OOD detection """
    # this is with precomputed logits
    # for every ood detection task we run all detectors. every detector is assigned a metric
    # the metric is reset before every  new detection task
    for task_name, test_inputs_loader in test_dataloaders.items():
        results = []
        id_data_name = task_name.split("_vs")[0]
        for detector_name, logit_detector in detectors.items():
            # print(f"INFO: fitting {detector_name} on {id_data_name}")
            _dest = f"/data/temporary/ivan/DeepDerma/OOD/results/{args.exp_code}"
            if args.testing:
                _dest = f"/data/temporary/ivan/DeepDerma/OOD/results/{args.exp_code}_testing" 
                os.makedirs(_dest,exist_ok=True)
            detector_label = f"{id_data_name}_{detector_name}"
            try:
                detectors[detector_name].get_fit_detector(data_loader = train_dataloaders[id_data_name],device = device,_filename =  _dest,detector_name = detector_label)
            except AttributeError as e:
                detectors[detector_name].fit(train_dataloaders[id_data_name],device=device)
                pass

        # for every detector we iterate over every dataloader : less context switches 
        if USE_LOGITS:
            logits_loader = precomputed_logits[task_name]
            print(f"--> {task_name} with logit-based ood detection")
            with torch.no_grad():
                for detector_name, logit_detector in logit_detectors.items():
                    print(detector_name)
                    for bag_logits, y in tqdm(logits_loader,total=len(logits_loader)):
                        outlier_scores = logit_detector.predict_features(bag_logits)
                        metric_calculators[detector_name].update(outlier_scores.to(device), y.to(device)) # add results to collection
                    
                    task_metrics = metric_calculators[detector_name].compute() # calculate metrics based on results
                    task_results =  {"Detector": detector_name, "Task": task_name} | task_metrics # concatenate dics
                    results.append(task_results)
                    metric_calculators[detector_name].reset()

            del logits_loader

        if USE_FEATS:
            print(f"--> {task_name} with feature-based ood detection")
            # for detector_name, feature_detector in feature_detectors.items():
            #     feature_detector.fit(test_inputs_loader,device=device)

            with torch.no_grad():
                # this is for detectors that require model inputs
                for detector_name, feature_detector in feature_detectors.items():
                    print(detector_name)
                    for x, y in tqdm(test_inputs_loader,total=len(test_inputs_loader)):
                        outlier_scores = feature_detector.predict(x.to(device)).view_as(y)
                        metric_calculators[detector_name].update(outlier_scores, y.to(device)) # add results to collection
                    
                    task_metrics = metric_calculators[detector_name].compute() # calculate metrics based on results
                    task_results =  {"Detector": detector_name, "Task": task_name} | task_metrics # concatenate dics
                    results.append(task_results)
                    metric_calculators[detector_name].reset()


        """ WRAP UP """
        df = pd.DataFrame(results)
        mean_scores = (
            df.groupby("Detector")[["AUROC", "AUTC", "AUPR-IN", "AUPR-OUT", "FPR95TPR"]].mean()
        )
        # mean_scores.sort_values(by=['AUROC'])
        filepath = f'/data/temporary/ivan/DeepDerma/OOD/results/{args.exp_code}/{task_name}-{datetime.now().strftime("%Y%m%d%H")}.csv'
        Path(filepath).parent.mkdir(parents=True, exist_ok=True) 
        print(df)
        print(mean_scores.sort_values("AUROC").to_csv(float_format="%.4f"))

        if args.testing:
            filepath = filepath.replace(args.exp_code,f"{args.exp_code}_testing")

        mean_scores.to_csv(filepath, float_format="%.4f")

def main(args):
    """Q: how do we organise the dataset creation?
    A: the commandline arg tasks will command the tasks. each task corresponds to a predefined set of arguments
    for creating a ood_detection_dataset, which will be implemented below"""
    """following https://pytorch-ood.readthedocs.io/en/latest/auto_examples/benchmarks/manual/cifar100_baseline.html#sphx-glr-auto-examples-benchmarks-manual-cifar100-baseline-py """
    
    
    """Q: what od/id labels do we assign?
    A: we assign the labels with the ood object ToUnknown()
    alternatively, we can assign labels <0 for od and >=0 for id. 
    Q: maybe it is better to add an argument to define od and id classes
    and then make 2 dataloaders with a target transform
    so: to the dataset, add a function get_id_dataloader(): which uses id classes
    and a function get_od_dataloader(): which uses od classes
    A: no. we can simply use the label dict for this that is used at the dataset creation.  

    following pytorch-ood the id have to be labeled >= 0, the od to (<0)"""

    run(args)

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
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--split', type=str, choices=["train",'val','test'], help='name of split')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'addmil'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--tasks', nargs="*")

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
parser.add_argument('--data_label_csv_path_train', type=str, default=None, help='data label directory') 
parser.add_argument('--data_label_csv_path_test', type=str, default=None, help='data label directory') 
parser.add_argument('--datatype', type=str, default="pt", help='data type',choices=["npy","h5","pt"])    
args = parser.parse_args()

# args.detectors = ["Entropy", "MSP", "EnergyBased", "MaxLogit","MCD", "KNN", "Mahalanobis", "RMD", "SHE", "TemperatureScaling", "KLMatching", "RMD", "ODIN"]
args.detectors = ["RMD", "KNN", "Mahalanobis"]

if args.data_label_csv_path_train is None :
    args.data_label_csv_path_train = args.data_label_csv_path_test

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

seed_torch(args.seed)

encoding_size = 1024
settings = {'split' : args.split,
            'tasks': args.tasks,
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
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Datasets')

args.n_classes = 2
test_datasets = []
train_datasets = {}
id_data_names = set([id_data_name.split("_vs")[0] for id_data_name in args.tasks])
for id_data_name in id_data_names:
    if id_data_name == "cobra":
        train_datasets[id_data_name] = Generic_WSI_OODDetection_Dataset(csv_path = args.data_label_csv_path_train , 
                                    data_dir= args.data_root_dir,
                                    shuffle = False, 
                                    seed = args.seed, 
                                    print_info = True,
                                    label_dict ={
                    "normal": 0,
                    "bcc": 1,
                } , # label cobra as 1 , other as -1
                                    patient_strat= False,
                                    ignore=[
                    "other_benign",
                    "dsn",
                    "sc",
                    "melanoma",
                    "adnexal_other",
                    "sa",
                    "mac",
                    "other",
                    "cylindrome",
                    "bccood",
                    "adnexal_carcinoma",
                    "metastasis",
                    "mcc",
                    "lymphoma",
                    "cobra2016-2020",
                    "scc"
                ],
                                    datatype=args.datatype,
                                    validate_inputs = True)
    else:
        raise NotImplementedError

for task in args.tasks:
    print(args.data_root_dir)
    if task == 'cobra_vs_scc':
        test_dataset = Generic_WSI_OODDetection_Dataset(csv_path = args.data_label_csv_path_test , 
                                data_dir= args.data_root_dir,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict ={
                "normal": 0,
                "bcc": 1,
                "scc": -1
            } , # label cobra as 1 , other as -1
                                patient_strat= False,
                                ignore=[
                "other_benign",
                "dsn",
                "sc",
                "melanoma",
                "adnexal_other",
                "sa",
                "mac",
                "other",
                "cylindrome",
                "bccood",
                "adnexal_carcinoma",
                "metastasis",
                "mcc",
                "lymphoma",
                "cobra2016-2020"
            ], # ignore everything but normal, cscc, bcc
                                datatype=args.datatype,
                                validate_inputs = True)
        

    elif task == 'cobra_vs_otherdiseases':
        test_dataset = Generic_WSI_OODDetection_Dataset(csv_path = args.data_label_csv_path_test , 
                                data_dir= args.data_root_dir,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict ={
                "normal": 0,
                "bcc": 1,
                "scc": -1,
                "adnexal_carcinoma": -2,
                "sa": -3,
                "lymphoma": -4,
                "mac": -5,
                "dsn": -6,
                "mcc": -7,
                "cylindrome": -8,
                "adnexal_other": -9,
                "melanoma": -10,
                "metastasis": -11,
                "sc": -12
            } , # label cobra as 1 , other as -1
                                patient_strat= False,
                                ignore=[
                "other_benign",
                "other",
                "cobra2016-2020",
                "bccood"
            ], # ignore everything but normal, cscc, bcc
                                datatype=args.datatype,
                                validate_inputs = True)
        
        

    else:
        raise NotImplementedError
    test_datasets.append(test_dataset)

if args.model_type in ['clam_sb', 'clam_mb']:
    assert args.subtyping 

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)



print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")


