import os
import numpy as np
import pandas as pd
import torch
import time
from scipy import stats
from dataset_modules.dataset_generic import Generic_MIL_Dataset
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from torch.utils.data import Dataset


def group_and_label_strat_kfold(df,groupby,label,n_splits,seed=42):
    # Initialize StratifiedGroupKFold
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Extract group IDs and their corresponding class labels for stratification
    groups = df[groupby].values
    strat_labels = df.groupby(groupby)[label].first().reindex(df[groupby]).values

    # Create folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, strat_labels, groups)):
        train_set = df.iloc[train_idx].reset_index(drop=True)
        val_set = df.iloc[val_idx].reset_index(drop=True)
        yield fold, train_set,val_set

class Multilabel_WSI_Dataset(Dataset):
	def __init__(self,
		csv_path = 'dataset_csv/ccrcc_clean.csv',
		shuffle = False, 
		seed = 7, 
		print_info = True,
		label_dicts = {},
		filter_dict = {},
		ignores={},
		patient_strat=False,
		slide_col = None,
		patient_voting = 'max',
		case_id_col = "case_id",
		datatype=None,
		validate_inputs  = False,
		data_dir = "",
		):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dicts (dict): Dictionary with key, value pairs for converting str labels to int
			ignores (list): List containing class labels to ignores
		"""
		self.label_dicts = label_dicts
		self.seed = seed
		self.datatype = datatype
		self.data_dir = data_dir
		self.label_cols = list(label_dicts.keys())
		self.ignores =ignores

		if not slide_col:
			slide_col = 'slide_id'
		self.slide_col = slide_col
		self.case_id_col = case_id_col
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.validate_inputs = validate_inputs

		slide_data = pd.read_csv(csv_path,dtype=str)
		slide_data = self.df_prep(slide_data)
		# change num classes
		self.num_classes = {k: len(set(label_dict.values())) for (k,label_dict) in self.label_dicts.items()}
		self.label_dicts_inv = {k: {v:k for (k,v) in label_dict.items()} for (k,label_dict) in self.label_dicts.items()}

		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)
		self.slide_data = slide_data

		if print_info:
			self.summarize()

	def summarize(self):
		print("label column: {}".format(self.label_cols))
		print("label dictionary: {}".format(self.label_dicts))
		print("number of classes: {}".format(self.num_classes))
		# for label_col in self.label_cols: 
		# 	print(label_col," ", self.slide_data[label_col].value_counts(sort = False))


	def df_prep(self,data, check_func=None):
		if self.slide_col != 'slide_id':
			data['slide_id'] = data[self.slide_col].copy()

		if self.case_id_col!= "case_id":
			data["case_id"] = data[self.case_id_col].copy()
			
		data.dropna(subset=['slide_id'], inplace=True)	

		for label_col,ignore in self.ignores.items():
			print("for colum ", label_col,"we remove ", ignore)
			mask = data[label_col].isin(ignore)
			data = data[~mask].reset_index(drop=True)


		for label_col, label_dict in self.label_dicts.items():
			data.dropna(subset=[label_col], inplace=True)	
			if label_dict is None:
				label_dict = {k:i for (i,k) in enumerate(set(data[label_col].values))}
				self.label_dicts[label_col] = label_dict
			elif len(label_dict) > 0:
				mask = data[label_col].isin(label_dict)
				data = data[mask].reset_index(drop=True)

			data[f"raw_{label_col}"] = data[label_col].copy()
			data[label_col] = data[label_col].apply(label_dict.get)


		data.reset_index(drop=True, inplace=True)
		if self.validate_inputs:
			print("validating inputs..")
			print("numbers before validating: ", len(data))
			mask = list(map(self.check_exists,data['slide_id'].tolist()))
			# mask =  data['slide_id'].apply(lambda x: self.check_exists(x))
			data[~mask]['slide_id'].to_csv("/data/temporary/ivan/DeepDerma/OOD/splits/invalid_files.csv",index=False)
			data = data[mask]

			data.reset_index(drop=True, inplace=True)
			print("done validating inputs: ", len(data))

			
		return data
	

	def check_exists(self,full_path):
		return os.path.exists(full_path)


	def sample_df(self,n_per_class,sample_column='label',inplace=False):
		if inplace:
			self.slide_data = self.slide_data.groupby(sample_column, group_keys=False).apply(lambda x: x.sample(n=min(n_per_class, len(x)), random_state=42)).reset_index(drop=True)
		else:
			return self.slide_data.groupby(sample_column, group_keys=False).apply(lambda x: x.sample(n=min(n_per_class, len(x)), random_state=42)).reset_index(drop=True)

class Multilabel_MIL_Dataset(Multilabel_WSI_Dataset):
	def __init__(self,
		data_dir, 
		**kwargs):
	
		super(Generic_MIL_Dataset, self).__init__(**kwargs)
		self.data_dir = data_dir
		self.use_h5 = False

	def load_from_h5(self, toggle):
		self.use_h5 = toggle


	def __getitem__(self, idx):
		try:
			slide_id = self.slide_data['slide_id'][idx]
			row = self.slide_data.iloc[idx]
			labels = row[self.label_cols].values.astype(int)

			
			if self.datatype:
				datatype = self.datatype
			else:
				datatype = self.datatypes[idx]

			if type(self.data_dir) == dict:
				source = self.slide_data['source'][idx]
				data_dir = self.data_dir[source]
			else:
				data_dir = self.data_dir


			if datatype == "pt":
				full_path = os.path.join(data_dir, slide_id)
				features = torch.load(full_path,map_location='cpu')
				return features, labels
				

			elif datatype == "npy":
				full_path = os.path.join(data_dir, slide_id)
				features = np.load(full_path)
				features = torch.from_numpy(features)
				return features, labels


			elif datatype == "h5":
				full_path = os.path.join(data_dir, slide_id)
				with h5py.File(full_path,'r') as hdf5_file:
					features = hdf5_file['features'][:]
					coords = hdf5_file['coords'][:]

				features = torch.from_numpy(features)
				return features, labels, coords
			
		except Exception as e:
			print(e)
			print(self.data_dir, slide_id, self.datatype)
			print(self.slide_data.iloc[0]["slide_id"].split(".")[-1])


class Multilabel_Split(Multilabel_MIL_Dataset):
	def __init__(self, slide_data, data_dir="", label_cols = None, label_dicts=None, num_classes=2,datatype=None,dict_is_part_of_superset=False):
		self.use_h5 = False
		self.slide_data = slide_data
		self.data_dir = data_dir
		self.num_classes = num_classes
		self.label_cols = label_cols
		self.datatypes = self.slide_data["slide_id"].apply(lambda x: x.split(".")[-1]).values
		self.datatype = datatype
		self.label_dicts = label_dicts.copy()
		# self.slide_cls_ids = [[] for i in range(self.num_classes)]
		# for i in range(self.num_classes):
		# 	self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

		if not dict_is_part_of_superset:
			for _d in self.label_dicts.keys():
				if len(set(self.label_dicts[_d].keys())) > len(set(self.slide_data[_d].values)):
					if _d == "label":
						print(f"Split inherited less values for column {_d} but NOT updating values and dict")
					new_domain_dict = {k:n for n,k in enumerate(set(self.slide_data[_d].values))}
					self.label_dicts[_d] = {k: new_domain_dict[v] for (k,v) in self.label_dicts[_d].items() if v in new_domain_dict.keys()}
					self.slide_data[_d] = self.slide_data[_d].map(new_domain_dict)
					print(f"Split inherited less values for column {_d}.", self.label_dicts[_d])
		

	def __len__(self):
		return len(self.slide_data)
		