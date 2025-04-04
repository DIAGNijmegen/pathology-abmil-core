import os
import sys
import argparse

from utils.ood_detection import precompute_logits, prepare_ood_detectors, aggregrate_patch_features,features_to_TSNE, RAW_FEATURE_SPACE, TRANSFORMED_FEATURE_SPACE, ATTENDED_FEATURE_SPACE, INVATTENDED_FEATURE_SPACE,INVMAXNORMATTENDED_FEATURE_SPACE
from utils.utils import *
from utils.core_utils import get_pretrained_model

from dataset_modules.dataset_ood import Generic_WSI_OODDetection_Dataset

import torch


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
    seed_torch(args.seed)
    print(f"INFO: Evaluating classifier on {args.tasks} datasets.")
    model = get_pretrained_model(args,fold=0,for_ood=True,device=device)
    """ Prepare the required components """

    dataloaders  = {}
    for task_name, task_dataset in zip(args.tasks,datasets):
        dataloaders[task_name] = get_split_loader(task_dataset, training=False,testing = args.testing)


    """ Run OOD detection """
    with torch.no_grad():
        # this is with precomputed logits
        # for every ood detection task we run all detectors. every detector is assigned a metric
        # the metric is reset before every  new detection task
        for (task_name, inputs_loader),dataset in zip(dataloaders.items(),datasets):
            filenames = dataset.patient_data['case_id'][:len(inputs_loader)]
            
            features_to_TSNE(loader = inputs_loader, filenames = filenames, 
                            model = RAW_FEATURE_SPACE(model),
                            task = task_name,experiment = args.exp_code, 
                            label_dict = {v:k for (k,v) in dataset.label_dict.items()},split = args.split, extra_label = "raw_mean")
            
            features_to_TSNE(loader = inputs_loader, filenames = filenames, 
                            model = INVATTENDED_FEATURE_SPACE(model),
                            task = task_name,experiment = args.exp_code, 
                            label_dict = {v:k for (k,v) in dataset.label_dict.items()},split = args.split, extra_label = "attention_inv")   

            features_to_TSNE(loader = inputs_loader, filenames = filenames, 
                            model = INVMAXNORMATTENDED_FEATURE_SPACE(model),
                            task = task_name,experiment = args.exp_code, 
                            label_dict = {v:k for (k,v) in dataset.label_dict.items()},split = args.split, extra_label = "attention_invmaxnorm")      
                 
            features_to_TSNE(loader = inputs_loader, filenames = filenames, 
                            model = ATTENDED_FEATURE_SPACE(model),
                            task = task_name,experiment = args.exp_code, 
                            label_dict = {v:k for (k,v) in dataset.label_dict.items()},split = args.split, extra_label = "transformed_attention")
            
            features_to_TSNE(loader = inputs_loader, filenames = filenames, 
                            model = TRANSFORMED_FEATURE_SPACE(model),
                            task = task_name,experiment = args.exp_code, 
                            label_dict = {v:k for (k,v) in dataset.label_dict.items()},split = args.split, extra_label = "transformed_mean")
            

            
            

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



# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--split', type=str, default=10, help='number of folds (default: 10)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'addmil'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--tasks',nargs='*')
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
args = parser.parse_args()

args.detectors = ["Entropy", "MSP", "EnergyBased", "MaxLogit","MCD", "KNN", "Mahalanobis", "RMD", "SHE", "TemperatureScaling", "KLMatching", "RMD", "ODIN"]

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_torch(args.seed)

encoding_size = 1024
settings = {'split' : args.split,
            'tasks': args.tasks,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'experiment': args.exp_code,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            }


if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Datasets')
if args.model_type in ['clam_sb', 'clam_mb']:
    assert args.subtyping 
            
datasets = []
for task in args.tasks:
    print(args.data_root_dir)
    if task == 'cobra_vs_scc':
        args.n_classes=2
        dataset = Generic_WSI_OODDetection_Dataset(csv_path = args.data_label_csv_path , 
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

        if args.model_type in ['clam_sb', 'clam_mb']:
            assert args.subtyping 

    elif task == 'bcc_vs_scc':
        args.n_classes=2
        dataset = Generic_WSI_OODDetection_Dataset(csv_path = args.data_label_csv_path , 
                                data_dir= args.data_root_dir,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict ={
                "bcc": 0,
                "scc": -1
            } , # label cobra as 1 , other as -1
                                patient_strat= False,
                                ignore=[
                "other_benign",
                "normal",
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
        args.n_classes=2
        dataset = Generic_WSI_OODDetection_Dataset(csv_path = args.data_label_csv_path , 
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
    datasets.append(dataset)

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