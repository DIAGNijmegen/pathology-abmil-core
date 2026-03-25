import pandas as pd
import numpy as np
import os
from dataset_modules.dataset_multilabel import Multilabel_Split
from dataset_modules.dataset import WSI_DATASET
from dataset_modules.dataset_generic import Generic_Split, Generic_MIL_Dataset, Generic_WSI_Classification_Dataset
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

def redistribute_domains_over_splits(datasets,domain_column="compound",label_column="label",test_size=0.2):
    # take all categories for domain_column in both sets, and redistributes such that the two sets have about equal values of both domains
    full_df = []
    for ds in datasets:
        full_df.append(ds.slide_data)
    first_dataset = datasets[0]
    full_df = pd.concat(full_df).reset_index(drop=True)

    stratify_col = full_df[[domain_column, label_column]].astype(str).agg('__'.join, axis=1)
    stratify_counts = stratify_col.value_counts()

    # Keep only strata with >=2 samples
    valid_strata = stratify_counts[stratify_counts >= 2].index
    mask = stratify_col.isin(valid_strata)

    df_valid = full_df[mask].copy()
    stratify_col_valid = stratify_col[mask]

    df_train, df_test = train_test_split(
        df_valid,
        test_size=test_size,
        stratify=stratify_col_valid,
        random_state=4521
    )
    df_leftover = full_df[~mask]
    if len(df_leftover) == 1:
        df_train = pd.concat([df_train,df_leftover])
    elif len(df_leftover) > 1:
        df_train_leftover, df_test_leftover = train_test_split(
            df_leftover,
            test_size=test_size,
            stratify=domain_column,
            random_state=4521
        )
        df_train = pd.concat([df_train,df_train_leftover])
        df_test = pd.concat([df_test,df_test_leftover])
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    split_train = Multilabel_Split(df_train, data_dir=first_dataset.data_dir, num_classes=first_dataset.num_classes,label_cols = first_dataset.label_cols, label_dicts=first_dataset.label_dicts,datatype=first_dataset.datatype,dict_is_part_of_superset=True)
    split_test = Multilabel_Split(df_test, data_dir=first_dataset.data_dir, num_classes=first_dataset.num_classes,label_cols = first_dataset.label_cols, label_dicts=first_dataset.label_dicts,datatype=first_dataset.datatype,dict_is_part_of_superset=True)

    return (split_train,split_test)


def select_column_values(DATASET,column_name,values,inplace=True):
    df_reduced = DATASET.slide_data[DATASET.slide_data[column_name].isin(values)].reset_index(drop=True)
    if inplace:
        DATASET.slide_data = df_reduced
    else:
        return df_reduced
    

def fewer_domains(MULTILABEL_DATASET,n_domains,domain_column="compound",inplace=False):
    top_n_groups = MULTILABEL_DATASET.slide_data[domain_column].value_counts().nlargest(n_domains).index
    # Filter the DataFrame to include only those top n groups
    df_reduced = MULTILABEL_DATASET.slide_data[MULTILABEL_DATASET.slide_data[domain_column].isin(top_n_groups)].reset_index(drop=True)
    print(f"Requested {n_domains} . Selected {len(top_n_groups)}")
    print(f"For {min(n_domains,len(top_n_groups))} domains of {domain_column} we selected {len(df_reduced)} rows")
    if len(set(MULTILABEL_DATASET.label_dicts[domain_column].keys())) > len(set(df_reduced[domain_column].values)):
        new_domain_dict = {k:n for n,k in enumerate(set(df_reduced[domain_column].values))}
        MULTILABEL_DATASET.label_dicts[domain_column] = {k: new_domain_dict[v] for (k,v) in MULTILABEL_DATASET.label_dicts[domain_column].items() if v in new_domain_dict.keys()}
        df_reduced[domain_column] = df_reduced[domain_column].map(new_domain_dict)
        print(f"Selected fewer domains. updated label dict: column {domain_column}", MULTILABEL_DATASET.label_dicts[domain_column])
    
    if inplace:
        MULTILABEL_DATASET.slide_data = df_reduced
    else:
        return df_reduced


def task_to_dataset(args):
    print("dataset args: ")
    print(args)
    if args.task == 'task_1_tumor_vs_normal':
        args.n_classes=2
        dataset = WSI_DATASET(csv_path = args.data_label_csv_path,
                                data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                                shuffle = False, 
                                slide_col=args.slide_col,
                                label_col= 'label',
                                validate_inputs=False,
                                datatype="pt",
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                                patient_strat=False,
                                ignore=[])

    elif args.task == 'cscc_vs_noncscc':
        args.n_classes=2
        dataset = WSI_DATASET(csv_path = args.data_label_csv_path,
                                data_dir= os.path.join(args.data_root_dir, 'features'),
                                shuffle = False, 
                                slide_col=args.slide_col,
                                label_col= 'label',
                                validate_inputs=False,
                                datatype="pt",
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'non-cscc':0, 'cscc':1},
                                patient_strat=False,
                                ignore=[])

    elif args.task == 'task_2_tumor_subtyping':
        args.n_classes=3
        dataset = WSI_DATASET(csv_path = args.data_label_csv_path,
                                data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                                shuffle = False, 
                                slide_col=args.slide_col,
                                label_col= 'label',
                                validate_inputs=False,
                                datatype="pt",
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                                patient_strat= False,
                                ignore=[])

        if args.model_type in ['clam_sb', 'clam_mb']:
            assert args.subtyping 

    elif args.task == 'bcc_bin':
        args.n_classes=2
        dataset = WSI_DATASET(csv_path = args.data_label_csv_path , 
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
        dataset = WSI_DATASET(csv_path = args.data_label_csv_path , 
                                data_dir= args.data_root_dir,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = {'normal': 0, 'bcc': 1},
                                patient_strat= False,
                                slide_col= args.slide_col,
                                label_col = args.label_col,
                                validate_inputs = False,
                                ignore=[],
                                datatype=args.datatype)

    elif args.task == 'abnormal_vs_normal':
        args.n_classes=2

        dataset = WSI_DATASET(
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
		
def get_merged_split_from_df(DATASET, all_splits, split_keys=['train'], multilabel=False):

    merged_split = []
    for split_key in split_keys:
        split = all_splits[split_key]
        if len(split) > 0:
            split = split.dropna().reset_index(drop=True).tolist()
            merged_split.extend(split)
    if len(merged_split) > 0:
        mask = DATASET.slide_data['case_id'].isin(merged_split)
        df_slice = DATASET.slide_data[mask].reset_index(drop=True)
        if multilabel:
            split = Multilabel_Split(df_slice, data_dir=DATASET.data_dir, num_classes=DATASET.num_classes,label_cols = DATASET.label_cols, label_dicts=DATASET.label_dicts,datatype=DATASET.datatype)
        else:
            split = Generic_Split(df_slice, data_dir="", num_classes=DATASET.num_classes,datatype=DATASET.datatype)
    else:
        split = None
    
    return split

def return_merged_split(DATASET,csv_path,splits=[],multilabel=False):
    all_splits = pd.read_csv(csv_path, dtype=DATASET.slide_data['slide_id'].dtype)  # Without "dtype=DATASET.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to DATASET.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
    split = get_merged_split_from_df(DATASET,all_splits=all_splits,split_keys=splits,multilabel=multilabel)
    return split

def get_fewshot_split(DATASET,csv_path,split_key="test",n_per_group=5,group_column="domain",class_column="label",multilabel=False):
    if multilabel:
        SPLITCLASS = Multilabel_Split
    else:
        SPLITCLASS = None
    all_splits = pd.read_csv(csv_path, dtype=DATASET.slide_data['slide_id'].dtype)  # Without "dtype=DATASET.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to DATASET.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.

    split = all_splits[split_key]
    split = split.dropna().reset_index(drop=True)
    if len(split) > 0:
        mask = DATASET.slide_data['slide_id'].isin(split.tolist())
        df_slice = DATASET.slide_data[mask].reset_index(drop=True).copy()
        # fewshot_slice =  df_slice.groupby(group_column, group_keys=False).apply(lambda x: x.sample(n=min(n_per_group, len(x)), random_state=42))
        fewshot_slice = (
            df_slice
            .groupby([group_column, class_column], group_keys=False)
            .apply(lambda x: x.sample(n=min(n_per_group, len(x)), random_state=42))
            .reset_index(drop=True)
            .copy()
        )
        fewshot_slice = fewshot_slice.reset_index(drop=True).copy()

        test_slice = pd.concat([df_slice, fewshot_slice]).drop_duplicates(keep=False).reset_index(drop=True).copy()
        if multilabel:
            fewshot_split = Multilabel_Split(fewshot_slice, data_dir=DATASET.data_dir, num_classes=DATASET.num_classes,label_cols = DATASET.label_cols, label_dicts=DATASET.label_dicts, datatype=DATASET.datatype)
            test_split = Multilabel_Split(test_slice, data_dir=DATASET.data_dir, num_classes=DATASET.num_classes,label_cols = DATASET.label_cols, label_dicts=DATASET.label_dicts, datatype=DATASET.datatype)
        else:
            fewshot_split =  Generic_Split(fewshot_slice, data_dir=DATASET.data_dir, num_classes=DATASET.num_classes,datatype=DATASET.datatype)
            test_split = Generic_Split(test_slice, data_dir=DATASET.data_dir, num_classes=DATASET.num_classes,datatype=DATASET.datatype)
    return fewshot_split, test_split



def return_custom_splits(DATASET, csv_path=None,merge_id_and_ood_splits=False,multilabel=False):
    # this should return:
    # train,val, test split. where val split is 
    assert csv_path 
    if not csv_path:
        if multilabel:
            return Multilabel_Split(DATASET.slide_data, data_dir=DATASET.data_dir, num_classes=DATASET.num_classes,label_cols = DATASET.label_cols, label_dicts=DATASET.label_dicts,datatype=DATASET.datatype)
        else:
            return Generic_Split(DATASET.slide_data, data_dir=DATASET.data_dir, num_classes=DATASET.num_classes,datatype=DATASET.datatype)
    all_splits = pd.read_csv(csv_path, dtype=DATASET.slide_data['slide_id'].dtype)  # Without "dtype=DATASET.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to DATASET.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
    split_names = list(filter(lambda x: "Unnamed" not in x, all_splits.columns))
    splits = {split_name:get_split_from_df(DATASET,all_splits,split_name,multilabel=multilabel) for split_name in split_names}
    return splits


def return_splits(DATASET, csv_path=None,merge_id_and_ood_splits=False,multilabel=False):
    # this should return:
    # train,val, test split. where val split is 
    assert csv_path 
    if not csv_path:
        if multilabel:
            return Multilabel_Split(DATASET.slide_data, data_dir=DATASET.data_dir, num_classes=DATASET.num_classes,label_cols = DATASET.label_cols, label_dicts=DATASET.label_dicts,datatype=DATASET.datatype)
        else:
            return Generic_Split(DATASET.slide_data, data_dir=DATASET.data_dir, num_classes=DATASET.num_classes,datatype=DATASET.datatype)
    all_splits = pd.read_csv(csv_path, dtype=DATASET.slide_data['slide_id'].dtype)  # Without "dtype=DATASET.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to DATASET.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
    train_split = get_split_from_df(all_splits, 'train',multilabel=multilabel)

    if merge_id_and_ood_splits:
        # merge train and test ood splits into 1 split
        val_split = get_merged_split_from_df(all_splits, ['val','ood-val'],multilabel=multilabel)
        test_split = get_merged_split_from_df(all_splits, ['test','ood-test'],multilabel=multilabel)
        return train_split,val_split,test_split
    else:
        val_split = get_split_from_df(all_splits, 'val',multilabel=multilabel)
        test_split = get_split_from_df(all_splits, 'test',multilabel=multilabel)
        if "ood-test" in all_splits.columns:				# merge train and test ood splits into 1 split
            test_ood_split = get_split_from_df(all_splits, 'ood-test',multilabel=multilabel)
            val_ood_split = get_split_from_df(all_splits, 'ood-val',multilabel=multilabel)
            return train_split, val_split, test_split, test_ood_split, val_ood_split
        else:
            return train_split, val_split,test_split

def get_split_from_df(DATASET, all_splits, split_key='train',multilabel=False):
    if type(all_splits) == str:
        all_splits = pd.read_csv(all_splits)
    split = all_splits[split_key].copy()
    split = split.dropna().reset_index(drop=True)
    if len(split) > 0:
        mask = DATASET.slide_data['case_id'].isin(split.tolist())
        df_slice = DATASET.slide_data[mask].reset_index(drop=True)

    else:
        df_slice = DATASET.slide_data.iloc[:0].copy()
        
    if multilabel:
        split = Multilabel_Split(df_slice, data_dir=DATASET.data_dir, num_classes=DATASET.num_classes,label_cols = DATASET.label_cols, label_dicts=DATASET.label_dicts,datatype=DATASET.datatype)  
    else:
        split = Generic_Split(df_slice, data_dir=DATASET.data_dir, num_classes=DATASET.num_classes,datatype=DATASET.datatype)
    
    return split

