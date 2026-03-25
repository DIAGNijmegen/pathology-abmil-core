import os
import numpy as np
import pandas as pd
from scipy import stats
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from torch.utils.data import Dataset

# cyclegan data


split_df = pd.read_csv("/data/temporary/ivan/DeepDerma/documents/classifier_splits/mohs_based_on_2024_test/splits_0.csv")
split_df = split_df.loc[:, ~split_df.columns.str.contains('Unnamed')]
ids_in_splits = split_df.values.ravel().tolist()
ids_in_splits = [x for x in ids_in_splits if not pd.isna(x)]

class WSI_DATASET(Dataset):
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
		slide_col = None,
		patient_voting = 'max',
		datatype="",
		validate_inputs  = False,
		# slide_dir = ,
		data_dir = "",
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
		# self.slide_dir = slide_dir
		self.data_dir = data_dir
		if not label_col:
			label_col = 'label'
		self.label_col = label_col
		if not slide_col:
			slide_col = 'slide_id'
		self.slide_col = slide_col
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.validate_inputs = validate_inputs

		slide_data = pd.read_csv(csv_path,dtype=str)
		slide_data = self.filter_df(slide_data, filter_dict)

		slide_data = self.df_prep(slide_data, self.label_dict, ignore)
		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)
		self.slide_data = slide_data.copy()

		self.patient_data_prep(patient_voting)
		self.cls_ids_prep()

		if print_info:
			self.summarize()
		# print("after init: ", [ f for f in ids_in_splits if f not in self.slide_data["case_id"].values])

	def summarize(self):
		print("label column: {}".format(self.label_col))
		print("label dictionary: {}".format(self.label_dict))
		print("number of classes: {}".format(self.num_classes))
		print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
		for  i in self.class_ids:
			print('Patient-LVL; Number of samples registered in class %s: %d' % (self.label_dict_inv[i], self.patient_cls_ids[i].shape[0]))
			print('Slide-LVL; Number of samples registered in class %s: %d' % (self.label_dict_inv[i], self.slide_cls_ids[i].shape[0]))


	def filter_df(self, df, filter_dict={}):
		if len(filter_dict) > 0:
			filter_mask = np.full(len(df), True, bool)
			# assert 'label' not in filter_dict.keys()
			for key, val in filter_dict.items():
				mask = df[key].isin(val)
				filter_mask = np.logical_and(filter_mask, mask)
			df = df[filter_mask]
		return df

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

	def df_prep(self,data, label_dict, ignore, check_func=None):
		print("before processing df: ", len(data))
		if self.label_col != 'label':
			data['label'] = data[self.label_col].copy()
		else:
			data['raw_label'] = data['label'].copy()

		if self.data_dir:
			data[self.slide_col] = data[self.slide_col].apply(lambda x: os.path.join(self.data_dir, x + "." + self.datatype))
		if self.slide_col != 'slide_id':
			data['slide_id'] = data[self.slide_col].copy()
		mask = data['label'].isin(ignore)
		data = data[~mask]
		if len(self.label_dict) > 0:
			mask = data['label'].isin(label_dict.keys())
			data = data[mask]
		data.reset_index(drop=True, inplace=True)
		data.dropna(subset=['slide_id'], inplace=True)	
		if self.validate_inputs:
			print("validating inputs..")
			print("numbers before validating: ", len(data))
			# multiprocessing:
			paths = data['slide_id'].tolist()
			# with Pool(processes=cpu_count()) as pool:
			# 	mask = list(tqdm(pool.imap_unordered(self.check_exists, paths), total=len(paths)))
			mask = list(map(self.check_exists,paths))
			data = data[mask]
			data.reset_index(drop=True, inplace=True)
			print("done validating inputs")
			print("after validating: ", len(data))
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

	def check_exists(self,full_path):
		
		return os.path.exists(full_path)

	# def get_merged_split_from_df(self, all_splits, split_keys=['train']):
	# 	merged_split = []
	# 	for split_key in split_keys:
	# 		split = all_splits[split_key]
	# 		if len(split) > 0:
	# 			split = split.dropna().reset_index(drop=True).tolist()
	# 			merged_split.extend(split)
	# 	if len(merged_split) > 0:
	# 		mask = self.slide_data['case_id'].isin(merged_split)
	# 		df_slice = self.slide_data[mask].reset_index(drop=True)
	# 		split = Generic_Split(df_slice, data_dir="", num_classes=self.num_classes,datatype=self.datatype)
	# 	else:
	# 		split = None
		
	# 	return split
	
	# def merge_splits(self, split_objects):
	# 	slide_data = pd.concat([s.slide_data for s in split_objects], ignore_index=True)
	# 	split = Generic_Split(slide_data, data_dir="", num_classes=self.num_classes,datatype=self.datatype)
	# 	return split

	# def return_splits(self, csv_path=None,return_ood_splits=False,merge_id_and_ood_splits=False):
	# 	# this should return:
	# 	# train,val, test split. where val split is 
	# 	if not csv_path:
	# 		return Generic_Split(self.slide_data, data_dir="", num_classes=self.num_classes,datatype=self.datatype)
		
	# 	all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.

	# 	train_split = self.get_split_from_df(all_splits, 'train')

	# 	if merge_id_and_ood_splits:
	# 		# merge train and test ood-train, ood-test splits into 1 split
	# 		val_split = self.get_merged_split_from_df(all_splits, ['val','ood-val'])
	# 		test_split = self.get_merged_split_from_df(all_splits, ['test','ood-test'])
	# 		return train_split,val_split,test_split
	# 	else:
	# 		val_split = self.get_split_from_df(all_splits, 'val')
	# 		test_split = self.get_split_from_df(all_splits, 'test')
	# 		if return_ood_splits:				# also return ood splits
	# 			if not 'ood-test' in all_splits.columns:
	# 				raise ValueError("OOD split not defined")
	# 			test_ood_split = self.get_split_from_df(all_splits, 'ood-test')
	# 			val_ood_split = self.get_split_from_df(all_splits, 'ood-val')
	# 			return train_split, val_split, test_split, test_ood_split, val_ood_split
	# 		else:
	# 			return train_split, val_split,test_split

	# def get_split_from_df(self, all_splits, split_key='train'):
	# 	split = all_splits[split_key]
	# 	split = split.dropna().reset_index(drop=True)
	# 	if len(split) > 0:
	# 		mask = self.slide_data['case_id'].isin(split.tolist())
	# 		df_slice = self.slide_data[mask].reset_index(drop=True)
	# 		split = Generic_Split(df_slice, data_dir="", num_classes=self.num_classes,datatype=self.datatype)
	# 	else:
	# 		df_slice = self.slide_data.iloc[:0].copy()
	# 		split = Generic_Split(df_slice, data_dir="", num_classes=self.num_classes,datatype=self.datatype)
		
	# 	return split
	