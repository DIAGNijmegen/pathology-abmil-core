import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedGroupKFold


from torch.utils.data import Dataset
from dataset_modules.dataset_generic import Generic_MIL_Dataset, Generic_Split
import h5py

from utils.utils import generate_split, nth

class WSI_OODDetection_Dataset(Generic_MIL_Dataset):
	def __init__(self,
		csv_path = 'dataset_csv/ccrcc_clean.csv',
		shuffle = False, 
		seed = 7, 
		print_info = True,
		label_dict = {},
		filter_dict = {},
		ignore=[],
		patient_strat=False,
		label_col = None,
		patient_voting = 'max',
		datatype="h5",
		validate_inputs  = False,
		data_dir = None
		):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dict (dict): Dictionary with key, value pairs for converting str labels to int
			ignore (list): List containing class labels to ignore
		"""
		self.label_dict = label_dict
		self.num_classes = len(set(self.label_dict.values()))
		self.class_ids = list(label_dict.values())
		self.label_dict_inv = {v:k for (k,v) in label_dict.items()}
		self.seed = seed
		self.datatype = datatype
		self.data_dir = data_dir
		if not label_col:
			label_col = 'label'
		self.label_col = label_col
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.validate_inputs = validate_inputs

		slide_data = pd.read_csv(csv_path)
		slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)
		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)
		self.slide_data = slide_data

		self.patient_data_prep(patient_voting)
		self.cls_ids_prep()

		if print_info:
			self.summarize()

	def summarize(self):
		print("label column: {}".format(self.label_col))
		print("label dictionary: {}".format(self.label_dict))
		print("number of classes: {}".format(self.num_classes))
		print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
		for  i in self.class_ids:
			print('Patient-LVL; Number of samples registered in class %s: %d' % (self.label_dict_inv[i], self.patient_cls_ids[i].shape[0]))
			print('Slide-LVL; Number of samples registered in class %s: %d' % (self.label_dict_inv[i], self.slide_cls_ids[i].shape[0]))

	def patient_data_prep(self, patient_voting='max'):
		patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
		patient_labels = []
		
		for p in patients:
			locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
			assert len(locations) > 0
			label = self.slide_data['label'][locations].values
			if patient_voting == 'max':
				label = label.max() # get patient label (MIL convention)
			elif patient_voting == 'maj':
				label = stats.mode(label)[0]
			else:
				raise NotImplementedError
			patient_labels.append(label)
		
		self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

	def df_prep(self,data, label_dict, ignore, label_col,check_func=None):
		if label_col != 'label':
			data['label'] = data[label_col].copy()

		mask = data['label'].isin(ignore)
		data = data[~mask]
		data.reset_index(drop=True, inplace=True)
		if self.validate_inputs:
			print("validating inputs..")
			mask =  data['slide_id'].apply(lambda x: self.check_exists(x))
			data = data[mask]
			data.reset_index(drop=True, inplace=True)

		for i in data.index:
			key = data.loc[i, 'label']
			data.at[i, 'label'] = label_dict[key]

		return data
	
	
	def cls_ids_prep(self):
		# store ids corresponding each class at the patient or case level
		self.patient_cls_ids = [[] for i in range(self.num_classes)]		
		for i in range(self.num_classes):
			self.patient_cls_ids[i] = np.where(self.patient_data['label'] == self.class_ids[i])[0]
		# store ids corresponding each class at the slide level

		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == self.class_ids[i])[0]

	def check_exists(self,case_id):
		if self.datatype == "pt":
			full_path = os.path.join(self.data_dir, '{}.pt'.format(case_id))

		elif self.datatype == "npy":
			full_path = os.path.join(self.data_dir, '{}.npy'.format(case_id))

		elif self.datatype == "h5":
			full_path = os.path.join(self.data_dir,'h5_files','{}.h5'.format(case_id))

		return os.path.exists(full_path)

	def get_merged_split_from_df(self, all_splits, split_keys=['train']):
		merged_split = []
		for split_key in split_keys:
			split = all_splits[split_key]
			split = split.dropna().reset_index(drop=True).tolist()
			merged_split.extend(split)

		if len(split) > 0:
			mask = self.slide_data['case_id'].isin(merged_split)
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes,datatype=self.datatype)
		else:
			split = None
		
		return split
	

	def return_splits(self, csv_path=None,merge_id_ood=False):
		# this should return:
		# train,val, test split. where val split is 
		assert csv_path 
		all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
		train_split = self.get_split_from_df(all_splits, 'train')
		if merge_id_ood:
			val_split = self.get_merged_split_from_df(all_splits, ['val','ood-val'])
			test_split = self.get_merged_split_from_df(all_splits, ['test','ood-test'])
			return train_split,val_split,test_split
		else:
			val_split = self.get_split_from_df(all_splits, 'val')
			test_split = self.get_split_from_df(all_splits, 'test')
			test_ood_split = self.get_split_from_df(all_splits, 'ood-test')
			val_ood_split = self.get_split_from_df(all_splits, 'ood-val')
			return train_split, val_split, test_split, test_ood_split, val_ood_split

	def get_split_from_df(self, all_splits, split_key='train'):
		split = all_splits[split_key]
		split = split.dropna().reset_index(drop=True)
		if len(split) > 0:
			mask = self.slide_data['case_id'].isin(split.tolist())
			df_slice = self.slide_data[mask].reset_index(drop=True)
			split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes,datatype=self.datatype)
		else:
			split = None
		
		return split
	