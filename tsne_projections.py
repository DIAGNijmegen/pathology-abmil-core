import os
import sys
import argparse
import torch

from utils.ood_detection import RAW_FEATURE_SPACE, TRANSFORMED_FEATURE_SPACE, ATTENDED_FEATURE_SPACE, INVATTENDED_FEATURE_SPACE,INVMAXNORMATTENDED_FEATURE_SPACE
from utils.projections import make_projections
from utils.core_utils import get_pretrained_model

from dataset_modules.dataset_ood import WSI_OODDetection_Dataset
from pathlib import Path
from utils.utils import get_split_loader
import numpy as np




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
    print(f"INFO: making projections of {args.tasks} datasets.")
    try:
        model = get_pretrained_model(args,fold=0,for_ood=True,device=device)
    except:
        print("not using a pretrained classifier")
        model = None
    """ Prepare the required components """
    test_dataloaders  = {}
    train_dataloaders = {}
    for task_name,task_dataset in zip(args.tasks,datasets):
        print(f"creating dataloader for dataset with {len(task_dataset.slide_data)} samples")
        try:
            train_split, val_split, test_split = task_dataset.return_splits(csv_path=Path(args.split_file),merge_id_ood_for_projection=True)
        except KeyError:
            train_split, val_split, test_split = task_dataset.return_splits(csv_path=Path(args.split_file),merge_id_ood_for_projection=False) 

        train_dataloaders[task_name] = get_split_loader(train_split,testing=args.testing, training=False)
        test_dataloaders[task_name] = get_split_loader(val_split,testing=args.testing, training=False)


    print("dataloaders are prepared")

    if args.testing:
        args.exp_code = f"{args.exp_code}_testing" 

    """ Run OOD detection """   
    with torch.no_grad():
        for (task_name, train_inputs),test_inputs,dataset in zip(train_dataloaders.items(),test_dataloaders.values(),datasets):
            label_dict = {v:k for (k,v) in dataset.label_dict.items()}
            make_projections(loader_A = train_inputs,loader_B=test_inputs, 
                            model = RAW_FEATURE_SPACE(model),
                            task = task_name, label_dict = label_dict, args = args,
                            split = "trainval", extra_label = "raw_mean")
            

            if model is not None:
                make_projections(loader_A = train_inputs,loader_B=test_inputs,
                                model = TRANSFORMED_FEATURE_SPACE(model),
                                task = task_name, label_dict = label_dict, args = args,
                                split = "trainval", extra_label = "transformed_mean")
                
                make_projections(loader_A = train_inputs,loader_B=test_inputs, 
                                model = ATTENDED_FEATURE_SPACE(model),
                                task = task_name, label_dict = label_dict, args = args,
                                split = "trainval", extra_label = "transformed_attention")

                make_projections(loader_A = train_inputs,loader_B=test_inputs, 
                                model = INVATTENDED_FEATURE_SPACE(model),
                                task = task_name, label_dict = label_dict, args = args,
                                split = "trainval", extra_label = "attention_inv")

                make_projections(loader_A = train_inputs,loader_B=test_inputs, 
                                model = INVMAXNORMATTENDED_FEATURE_SPACE(model),
                                task = task_name, label_dict = label_dict, args = args,
                                split = "trainval", extra_label = "attention_invmaxnorm")




            
            del train_inputs,test_inputs,dataset
            

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
parser.add_argument('--data_root_dir', type=str, default="", 
                    help='data directory')
# data options
parser.add_argument('--classifier_results_dir', default='', help='')
parser.add_argument('--projections_results_dir', default='', help='')
parser.add_argument('--exp_code', type=str, default=None, help='experiment code for saving results')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--tasks', nargs="*")
parser.add_argument('--split_file', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--fold_nr', type=int, default=0, help='the fold. dictates data fold and model trained on fold')
parser.add_argument('--label_col',type=str,default="label")
parser.add_argument('--slide_col',type=str,default="slide_id")

# model options
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'addmil'], default='addmil', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')

### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--B', type=int, default=8, help='number of positive/negative patches to sample for clam')
parser.add_argument('--data_label_csv_path', type=str, default=None, help='data label directory') 
parser.add_argument('--datatype', type=str, default=None, help='data type',choices=[None,"npy","h5","pt"])    
# plotting options
parser.add_argument('--opacity_label', type=str, default=None)
parser.add_argument("--class_tag",type=str,choices=["distribution","cluster","label","split","cohort","scanner"],default="label")
parser.add_argument("--color_tag",type=str,choices=["distribution","cluster","label","split","cohort","scanner"],default="label")
parser.add_argument("--symbol_tag",type=str,choices=["distribution","cluster","label","split","cohort","scanner"],default="distribution")
parser.add_argument("--border_tag",type=str,choices=["distribution","split","cohort","scanner"],default=None)



args = parser.parse_args()

if not args.split_file:
    args.split_file = f"{args.split_dir}_{args.fold_nr}.csv"

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
seed_torch(args.seed)
args.results_dir = args.projections_results_dir


print('\nLoad Datasets')
if args.model_type in ['clam_sb', 'clam_mb']:
    assert args.subtyping 
            
args.n_classes = 2

datasets = []
# id_data_names = set([id_data_name.split("_vs")[0] for id_data_name in args.tasks])
for task in args.tasks:
    print(args.data_root_dir)
    if task == 'new_scc_vs_old_scc':
        ood_detection_dataset = WSI_OODDetection_Dataset(csv_path= args.data_label_csv_path , 
                shuffle = False, 
                seed = args.seed, 
                print_info = False,
                slide_col = args.slide_col,
                label_col = args.label_col,
                label_dict = {
                    "scc": 0,
                    "cscc": -1,
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
                    "adnexal_carcinoma",
                    "metastasis",
                    "mcc",
                    "lymphoma",
                    "cobra2016-2020",
                    "bcc",
                    "bccood",
                    "normal"
                ], # ignore everything but normal, cscc, bcc
                datatype=args.datatype,
                validate_inputs = True)
        
    elif task == 'bcc_benign_vs_cscc':
        ood_detection_dataset = WSI_OODDetection_Dataset(csv_path= args.data_label_csv_path , 
                shuffle = False, 
                seed = args.seed, 
                print_info = False,
                slide_col = args.slide_col,
                label_col = args.label_col,
                label_dict = {
                    "normal": -1,
                    "bcc": -2,
                    "bccood": -3,
                    "scc": 0,
                    "cscc": 1,
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
                    "adnexal_carcinoma",
                    "metastasis",
                    "mcc",
                    "lymphoma",
                    "cobra2016-2020"
                ], # ignore everything but normal, cscc, bcc
                datatype=args.datatype,
                validate_inputs = True)
        
    elif task == 'cobra_vs_old_cscc':
        ood_detection_dataset = WSI_OODDetection_Dataset(csv_path = args.data_label_csv_path , 
                shuffle = False, 
                seed = args.seed, 
                print_info = False,
                slide_col = args.slide_col,
                label_col = args.label_col,
                label_dict ={
                    "normal": 0,
                    "bccood": 1, 
                    "bcc": 2,
                    "cscc":-13,
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
                ], # ignore everything but normal, cscc, bcc
                datatype=args.datatype,
                validate_inputs = True)  
          
    elif task == 'benign_bcc_vs_scc':
        ood_detection_dataset = WSI_OODDetection_Dataset(csv_path = args.data_label_csv_path , 
                shuffle = False, 
                seed = args.seed, 
                print_info = False,
                slide_col = args.slide_col,
                label_col = args.label_col,
                label_dict = {
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
                    "bccood"
                    "cobra2016-2020"
                ], # ignore everything but normal, cscc, bcc
                datatype=args.datatype,
                validate_inputs = True)
        
    elif task == 'cobra_vs_otherdiseases':
        ood_detection_dataset = WSI_OODDetection_Dataset(csv_path = args.data_label_csv_path , 
                shuffle = False, 
                seed = args.seed, 
                print_info = False,
                slide_col = args.slide_col,
                label_col = args.label_col,
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
                    "sc": -12,
                    'bccood': -13
                } , # label cobra as 1 , other as -1
                patient_strat= False,
                ignore=[
                    "other_benign",
                    "other",
                    "cobra2016-2020",
                ], # ignore everything but normal, cscc, bcc
                datatype=args.datatype,
                validate_inputs = True)
        
    elif task == 'cobra_and_oldbcc_vs_otherdiseases':
        ood_detection_dataset = WSI_OODDetection_Dataset(csv_path = args.data_label_csv_path , 
                shuffle = False, 
                seed = args.seed, 
                print_info = False,
                slide_col = args.slide_col,
                label_col = args.label_col,
                label_dict ={
                    "normal": 0,
                    "bccood": 1, 
                    "bcc": 2,
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
                ], # ignore everything but normal, cscc, bcc
                datatype=args.datatype,
                validate_inputs = True)

    elif task == 'tumor_patches_vs_no_tumor_patches':
        ood_detection_dataset = WSI_OODDetection_Dataset(csv_path= args.data_label_csv_path , 
                shuffle = False, 
                seed = args.seed, 
                print_info = False,
                slide_col = args.slide_col,
                label_col = args.label_col,
                label_dict = {
                    "cobra_benign": 0,
                    "scc_notumor": 1,
                    "cscc_notumor": 2,
                } , # label cobra as 1 , other as -1
                patient_strat= False,
                ignore=[
                ], # ignore everything but normal, cscc, bcc
                datatype=args.datatype,
                validate_inputs = True)


    elif task == 'scanner_scc_vs_cscc_vs_cobrabenign':
        ood_detection_dataset = WSI_OODDetection_Dataset(csv_path= args.data_label_csv_path , 
                shuffle = False, 
                seed = args.seed, 
                print_info = False,
                slide_col = args.slide_col,
                label_col = args.label_col,
                label_dict = {
                    "cobra_ood": 0,
                    "S1": 1,
                    "S2": 2,
                    "S3": 3,
                    "cobra": 4,
                } , # label cobra as 1 , other as -1
                patient_strat= False,
                ignore=[
                ], # ignore everything but normal, cscc, bcc
                datatype=args.datatype,
                validate_inputs = True)

    elif task == 'artifact_vs_normal':
        ood_detection_dataset = WSI_OODDetection_Dataset(csv_path= args.data_label_csv_path , 
                shuffle = False, 
                seed = args.seed, 
                print_info = False,
                slide_col = args.slide_col,
                label_col = args.label_col,
                label_dict = {
                    "0": 0,
                    "1": 1,
                } , # label cobra as 1 , other as -1
                patient_strat= False,
                ignore=[
                ], # ignore everything but normal, cscc, bcc
                datatype=args.datatype,
                validate_inputs = True)
    else:
        raise NotImplementedError
    datasets.append(ood_detection_dataset )

args.results_dir = os.path.join(args.results_dir, str(args.exp_code))

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)



    

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")