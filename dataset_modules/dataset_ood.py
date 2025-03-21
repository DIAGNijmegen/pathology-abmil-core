import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats

from torch.utils.data import Dataset
import h5py

from utils.utils import generate_split, nth



class Generic_WSI_OODDetection_Dataset(Dataset):
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
		data_dir = None,
		validate_inputs  = False
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
		self.seed = seed
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.data_dir = data_dir
		self.datatype = datatype

		if not label_col:
			label_col = 'label'
		self.label_col = label_col

		slide_data = pd.read_csv(csv_path)
		slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col,validate_inputs)
		self.slide_data = slide_data

		self.patient_data_prep(patient_voting)
		self.cls_ids_prep()

		if print_info:
			self.summarize()

		self.use_h5 = False

	def cls_ids_prep(self):
		# store ids corresponding each class at the patient or case level
		self.patient_cls_ids = [[] for i in range(self.num_classes)]		
		for i in range(self.num_classes):
			self.patient_cls_ids[i] = np.where(self.patient_data['label'] == self.class_ids[i])[0]
		# store ids corresponding each class at the slide level

		self.slide_cls_ids = [[] for i in range(self.num_classes)]
		for i in range(self.num_classes):
			self.slide_cls_ids[i] = np.where(self.slide_data['label'] == self.class_ids[i])[0]


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


	def df_prep(self, data, label_dict, ignore, label_col,validate_inputs):
		if label_col != 'label':
			data['label'] = data[label_col].copy()

		mask = data['label'].isin(ignore)
		data = data[~mask]
		data.reset_index(drop=True, inplace=True)
		if validate_inputs:
			print("validating inputs..")
			mask =  data['case_id'].apply(lambda x: self.check_exists(x))
			data = data[mask]
		data.reset_index(drop=True, inplace=True)
		
		for i in data.index:
			key = data.loc[i, 'label']
			data.at[i, 'label'] = label_dict[key]

		return data

	def check_exists(self,case_id):
		if self.datatype == "pt":
			full_path = os.path.join(self.data_dir, '{}.pt'.format(case_id))

		elif self.datatype == "npy":
			full_path = os.path.join(self.data_dir, '{}.npy'.format(case_id))

		elif self.datatype == "h5":
			full_path = os.path.join(self.data_dir,'h5_files','{}.h5'.format(case_id))

		return os.path.exists(full_path)

	def __len__(self):
		return len(self.slide_data)

	def summarize(self):
		print("label column: {}".format(self.label_col))
		print("label dictionary: {}".format(self.label_dict))
		print("number of classes: {}".format(self.num_classes))
		print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
		for i in range(self.num_classes):
			print('Patient-LVL; Number of samples registered in class %d: %d' % (self.class_ids[i], self.patient_cls_ids[i].shape[0]))
			print('Slide-LVL; Number of samples registered in class %d: %d' % (self.class_ids[i], self.slide_cls_ids[i].shape[0]))

	def get_list(self, ids):
		return self.slide_data['case_id'][ids]

	def getlabel(self, ids):
		return self.slide_data['label'][ids]

	def __getitem__(self, idx):
		case_id = self.slide_data['case_id'][idx]
		label = self.slide_data['label'][idx]
		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir

		# print("==DEBUG== \n",self.slide_data.head(),data_dir,self.data_dir)

		if self.datatype == "pt":
			if self.data_dir:
				full_path = os.path.join(data_dir, '{}.pt'.format(case_id))
				features = torch.load(features)
				return features, label
			
			else:
				return case_id, label

		elif self.datatype == "npy":
			full_path = os.path.join(data_dir, '{}.npy'.format(case_id))
			features = np.load(full_path)
			features = torch.from_numpy(features)
			return features, label


		elif self.datatype == "h5":
			full_path = os.path.join(data_dir,'h5_files','{}.h5'.format(case_id))
			with h5py.File(full_path,'r') as hdf5_file:
				features = hdf5_file['features'][:]
				coords = hdf5_file['coords'][:]

			features = torch.from_numpy(features)
			return features, label, coords


