import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections
from torch.utils.data.distributed import DistributedSampler
import math
from collections.abc import Iterator
from typing import Optional, TypeVar
from typing import Optional, TypeVar, Sequence

import torch
import torch.distributed as dist

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

_T_co = TypeVar("_T_co", covariant=True)
class DistributedWeightedRandomSampler(Sampler[_T_co]):
    """Sampler that applies weighted random sampling across multiple distributed processes.
    This sampler combines the behavior of `WeightedRandomSampler` (sampling based on weights)
    and `DistributedSampler` (splitting data across multiple processes).
    Args:
        weights (sequence): A sequence of weights, not necessarily summing up to one.
        num_samples (int): Total number of samples across all processes.
        num_replicas (int, optional): Number of processes in distributed training.
        rank (int, optional): Rank of the current process within num_replicas.
        replacement (bool, optional): If True, samples are drawn with replacement.
        shuffle (bool, optional): If True, shuffle indices before applying weights.
        seed (int, optional): Random seed for reproducibility.
    Example:
        >>> import torch
        >>> from torch.utils.data import DataLoader
        >>> from torch.utils.data.distributed import DistributedSampler
        >>> dataset = list(range(10))
        >>> weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        >>> sampler = DistributedWeightedRandomSampler(weights, num_samples=5, num_replicas=2, rank=0)
        >>> loader = DataLoader(dataset, sampler=sampler, batch_size=2)
        >>> for epoch in range(start_epoch, n_epochs):
        >>>     sampler.set_epoch(epoch)
        >>>     for batch in loader:
        >>>         print(batch)  # Each process gets a subset of the sampled indices
    """

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        num_replicas: Optional[int] = None,
        rank: int = None,
        replacement: bool = True,
        shuffle: bool = True,
        seed: int = 0
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        if len(weights) == 0:
            raise ValueError("Weights must be a non-empty sequence.")
        if any(w < 0 for w in weights):
            raise ValueError("Weights must be non-negative.")

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples  # total
        self.num_replicas = num_replicas
        self.num_samples_per_proc = int(math.ceil(self.num_samples * 1.0 / self.num_replicas))  # per process
        self.replacement = replacement
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Optional shuffling indices and weights before sampling
        indices = torch.arange(len(self.weights))
        if self.shuffle:
            perm = torch.randperm(len(indices), generator=g)  # Generate permutation
            indices = indices[perm]  # Shuffle indices
            weights = self.weights[perm]  # Shuffle weights in the same order
        else:
            weights = self.weights

        # Perform weighted sampling
        sampled_indices = torch.multinomial(
            weights, self.num_samples, self.replacement, generator=g
        )

        # Map sampled indices back to original dataset indices
        sampled_indices = indices[sampled_indices]

        # Distribute samples across processes
        sampled_indices = sampled_indices[self.rank:self.num_samples_per_proc * self.num_replicas:self.num_replicas]
        if len(sampled_indices) != self.num_samples_per_proc:
            raise RuntimeError(
                f"Expected {self.num_samples_per_proc} samples per process, "
                f"but got {len(sampled_indices)}."
            )

        return iter(sampled_indices.tolist())

    def __len__(self) -> int:
        return self.num_samples_per_proc

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for deterministic shuffling."""
        self.epoch = epoch


class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor(np.array([item[1] for item in batch]))
	return [img, label]

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]


def get_simple_loader(dataset, batch_size=1, num_workers=1):
	kwargs = {'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 

def get_split_loader(split_dataset, training = False, testing = False, weighted = False,distributed=False,rank=None,world_size=None):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				if distributed:
					sampler = DistributedWeightedRandomSampler(weights, len(weights),num_replicas=world_size,rank=rank)
				else:
					sampler = WeightedRandomSampler(weights,len(weights))	
			else:
				if distributed:
					sampler = DistributedSampler(split_dataset,shuffle=True,num_replicas=world_size,rank=rank)
				else:
					sampler = RandomSampler(split_dataset)
		else:	
			if distributed:
				sampler = DistributedSampler(split_dataset,shuffle=False,num_replicas=world_size,rank=rank)	
			else:
				sampler = SequentialSampler(split_dataset)
	else:
		if not distributed:
			ids = np.random.choice(np.arange(len(split_dataset)), int(len(split_dataset)*0.1), replace = False)
			sampler = SubsetSequentialSampler(ids)
		else:
			split_dataset.slide_data = split_dataset.slide_data.iloc[ids].reset_index()
			sampler = DistributedSampler(split_dataset, num_replicas=world_size, rank=rank, shuffle=training)
	
	loader =  DataLoader(split_dataset, batch_size=1, sampler = sampler, collate_fn = collate_MIL, **kwargs)
	return loader

def get_optim(model, args):
	if args.opt == "adam":
		print("\noptimizer Adam with lr", args.lr, 'decay ', args.reg)
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == "adamw":
		print("\noptimizer Adam with lr", args.lr, 'decay ', args.reg)
		optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		print("\noptimizer sgd with lr", args.lr, 'decay ', args.reg)
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	indices = np.arange(samples).astype(int)
	
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

			remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
			all_val_ids.extend(val_ids)

			if custom_test_ids is None: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))    
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)

