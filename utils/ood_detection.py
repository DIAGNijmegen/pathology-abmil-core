
import os
from typing import Callable,  Optional,  Tuple
import logging
log = logging.getLogger(__name__)
from pathlib import Path

# internal imports
from utils.utils import *


# pytorch imports
import torch
from torch import Tensor, logsumexp
from torch.utils.data import DataLoader,  TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


import pandas as pd
import numpy as np
from tqdm import tqdm

# ood detection imports
from pytorch_ood.detector import EnergyBased, MaxSoftmax, MaxLogit, Entropy, ODIN, Mahalanobis, RMD, KLMatching, SHE, DICE, MCD, TemperatureScaling, OpenMax, KNN
from pytorch_ood.utils import OODMetrics, TensorBuffer, contains_unknown, is_known,  is_unknown
from bs4 import BeautifulSoup
import umap

from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
***REMOVED***

def make_branch_return_logits(attentionbranch,*args,**kwargs):
    class OODAttentionBranch(attentionbranch):
        def __init__(self, *args, **kwargs):
            super(OODAttentionBranch, self).__init__(*args, **kwargs)

        def forward(self, x):
            outputs =  super().forward(x)
            return outputs[0] # logits

    return OODAttentionBranch(*args, **kwargs)

def non_verbose_mcd(verbosemcd,*args,**kwargs):
    class SILENTMCD(verbosemcd):
        def __init__(self, *args, **kwargs):
            super(SILENTMCD, self).__init__(*args, **kwargs)

        def predict(self, x: Tensor) -> Tensor:
            """
            :param x: input
            :return: outlier score
            """
            if self.mode == "var":
                return MCD.run(self.model, x, self.n_samples, batch_norm=self.batch_norm)[1]

            return (
                -MCD.run_mean(self.model, x, self.n_samples, batch_norm=self.batch_norm)
                .max(dim=1)
                .values
            )

    return SILENTMCD(*args, **kwargs)

def make_saveable_detector(detector,*args, **kwargs):
    class SaveableDetector(detector):
        def __init__(self, *args, **kwargs):
            super(SaveableDetector, self).__init__(*args, **kwargs)
            try:
                # required for SHE detector
                self.model = self.backbone
            except:
                pass

        def get_fit_detector(self, data_loader, device,_filename,detector_name):
            _filename = Path(_filename) / Path(f"{detector_name}.pkl")
            try:
                model = self.model
                with open(_filename,"rb") as f:
                    "Loading pre-fit detector"
                    self.__dict__ = pickle.load(f)
                    print("detector loaded")
                self.model = model
            except:
                self.fit(data_loader,device=device)
                model = self.model
                self.model = None
                _dict = self.__dict__
                with open(_filename,"wb") as f:
                    "Saving pre-fit detector"
                    pickle.dump(_dict,f,pickle.HIGHEST_PROTOCOL)
                self.model = model

    return SaveableDetector(*args, **kwargs)

def fit_on_branch_outputs(detector,*args, **kwargs):
    # wrapper necessary when we fit a detector on CLAM logits/featurs
    class CLAMDetector(detector):
        def __init__(self, *args, **kwargs):
            super(CLAMDetector, self).__init__(*args, **kwargs)
            try:
                # required for SHE detector
                self.model = self.backbone
            except:
                pass
        
        def fit(self, data_loader, device):
            """
            Fit parameters of the multi variate gaussian.

            :param data_loader: dataset to fit on.
            :param device: device to use
            """
            if device is None:
                device = list(self.model.parameters())[0].device
                log.warning(f"No device given. Will use '{device}'.")

            if isinstance(self.model, torch.nn.Module):
                log.debug(f"Moving model to device {device}")
                self.model.to(device)

            # changes to normal detector implemented below:
            output, y = extract_features_clam(data_loader, self.model, device,disable=False)
            # if self.__class__.__bases__[0].__name__ in ("KNN" , "TemperatureScaling" ) :
            #     return self.fit_features(output,y)
            # else:
            #     return self.fit_features(output, y, device)
        
            self.fit_features(output,y,device)

        def get_fit_detector(self, data_loader, device,_filename,detector_name):
            _filename = Path(_filename) / Path(f"{detector_name}.pkl")
            try:
                model = self.model
                with open(_filename,"rb") as f:
                    print("loading ", _filename)
                    "Loading pre-fit detector"
                    self.__dict__ = pickle.load(f)
                    print("detector loaded")
                self.model = model
            except:
                print(f"INFO: fitting {detector_name}")
                self.fit(data_loader,device=device)
                model = self.model
                self.model = None
                _dict = self.__dict__
                with open(_filename,"wb") as f:
                    "Saving pre-fit detector"
                    pickle.dump(_dict,f,pickle.HIGHEST_PROTOCOL)
                self.model = model


        def fit_features(self,output,y,device):
            try:
                super().fit_features(output,y,device)
            except:
                super().fit_features(output,y)

    return CLAMDetector(*args, **kwargs)

def extract_features_clam(
    data_loader: DataLoader, model: Callable[[Tensor], Tensor], device: Optional[str],disable=False
        ) -> Tuple[Tensor, Tensor]:
    """
    Helper to extract outputs from model. Ignores OOD inputs.

    :param data_loader: dataset to extract from
    :param model: neural network to pass inputs to
    :param device: device used for calculations
    :return: Tuple with outputs and labels
    """
    # TODO: add option to buffer to GPU
    buffer = TensorBuffer()
    with torch.no_grad():
        for batch in tqdm(data_loader,total=len(data_loader),disable=disable,ncols=50):
            x, y = batch
            x = x.to(device)
            # for CLAM features we have to unsqueeze the first dimension 
            x = x.unsqueeze(0) # x : (1,N,F)
            y = y.to(device)
            known = is_known(y)
            if known.any():
                z = model(x[known])
                # if type(z) is tuple:
                #     z = z[0]
                z = z.view(known.sum(), -1)  # flatten
                buffer.append("embedding", z)
                buffer.append("label", y[known])
        if buffer.is_empty():
            raise ValueError("No ID instances in loader \n")
    z = buffer.get("embedding").to(device)
    y = buffer.get("label")
    return z, y

def inverse_att(softmax_values):
    log_probs = torch.log(softmax_values)
    log_probs =  log_probs - log_probs.max()  # Normalize by shifting
    inv = - log_probs
    inv  = F.softmax(inv)
    return inv


def aggregrate_patch_features(model,_input,transformed=False,pool_attention=False ,inverse=False,maxnorm=False):
    # x, att_raw, att = model._compute_attention(x)  # Extract patch features
    if transformed:
        x, att_raw, att = model(_input)
        # x: torch.Size([3130, 512]) attended_features: torch.Size([3130, 512]) pooled:  torch.Size([1, 512])
        # or : x: torch.Size([1,3130, 512]) attended_features: torch.Size([1,3130, 512]) pooled:  torch.Size([1, 512])
        if pool_attention:
            att = att.detach()
            if inverse and maxnorm:
                att = F.softmax(torch.max(att_raw, dim=1, keepdim=True)[0] - att_raw,dim=1)
            elif inverse:
                att =  F.softmax(-att_raw,dim=1)

            if att.ndim > 2:
                att = att.permute(1,2,0)
                x = torch.bmm(att,x)
            else:
                x = torch.mm(att,x)

    else:
        x = model(_input)

    dim = 0 if x.ndim == 2 else 1
    pooled_features = x.mean(dim=dim, keepdim= (dim) < 1)  # Shape: [1205, 512] -> [1, 512] (1 feature vector per image)
    return pooled_features

def precompute_feats(model,loader, device,args,transform=None):
    computed = False
    if transform:
        _transform = transform(model)
    if not computed:
        print("INFO: Computing feats")
        feats = []
        ys = []
        with torch.no_grad():  # Disable gradient computation
            for i, (x, y) in tqdm(enumerate(loader),total=len(loader),disable=False,ncols=50):
                # bag_logits, att_raw, results_dict = model(x.to(device))
                bag_feats = _transform(x.to(device))
                feats.append(bag_feats)
                ys.append(y)
        feats = torch.cat(feats, dim=0) 
        ys = torch.cat(ys,dim=0)

    dataset = TensorDataset(feats,ys)
    return DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
                    batch_sampler=None)


def precompute_logits(model,loader,loader_name, device,args,recompute=True):
    computed = False
    if not recompute:
        try :
            print("INFO: loading logits \n")
            ckpt = torch.load(os.path.join(args.results_dir, f"logits_ys_{loader_name}.pt"), weights_only=False)
            logits = ckpt["logits"]
            ys = ckpt["ys"]
            computed = True
        except Exception as e : 
            print("INFO: Failed loading logits")
            print(e)
            computed = False

    if not computed:
        print("INFO: Computing logits")
        logits = []
        ys = []
        with torch.no_grad():  # Disable gradient computation
            for i, (x, y) in tqdm(enumerate(loader),total=len(loader),disable=False,ncols=50):
                # bag_logits, att_raw, results_dict = model(x.to(device))
                bag_logits = model(x.to(device))
                logits.append(bag_logits)
                ys.append(y)

        logits = torch.cat(logits, dim=0) 
        ys = torch.cat(ys,dim=0)
        savedict = {"logits": logits, "ys": ys}
        torch.save(savedict,os.path.join(args.results_dir, f"logits_ys_{loader_name}.pt"))

    dataset = TensorDataset(logits,ys)
    return DataLoader(dataset, batch_size=10, shuffle=False, sampler=None,
                    batch_sampler=None)

def RAW_FEATURE_SPACE(model):
    return lambda x: aggregrate_patch_features(lambda y: y,x , transformed=False)
def TRANSFORMED_FEATURE_SPACE(model):
    return lambda x: aggregrate_patch_features(lambda y: model._compute_attention(y),x, transformed=True,pool_attention=False)
def ATTENDED_FEATURE_SPACE(model):
    return lambda x: aggregrate_patch_features(lambda y: model._compute_attention(y),x, transformed=True,pool_attention=True)
def INVATTENDED_FEATURE_SPACE(model):
    return lambda x: aggregrate_patch_features(lambda y: model._compute_attention(y),x, transformed=True,pool_attention=True,inverse=True)

def INVMAXNORMATTENDED_FEATURE_SPACE(model):
    return lambda x: aggregrate_patch_features(lambda y: model._compute_attention(y),x, transformed=True,pool_attention=True,inverse=True,maxnorm=True)

def prepare_ood_detectors(model,args):
    print("INFO: Creating OOD Detectors")
    logit_detectors = {}
    feature_detectors = {}
    model_detectors = {}

    # if a detector has to be fit, it should be fit on the training dataset
    # if we predict / evaluate a detector, it should be evaluated on the test set + all ood samples
    # a detector requires fit_on_branch_outputs in case fitting is required, to handle the outputs of the attentionbranch

    for detector_name in args.detectors:
        if detector_name == "Entropy":
            # Calculates entropy based on logits. Higher entropy == more uniformly distributed posteriors == larger uncertainty
            logit_detectors["Entropy"] = Entropy(model)
        elif detector_name == "MSP":
            # takes maximum softmax probability over the prediction classes as outlier score
            logit_detectors["MSP"] =  MaxSoftmax(model)
        elif detector_name == "EnergyBased":
            # takes the negative energy of a vector of logits as outlier score
            logit_detectors["EnergyBased"] =  EnergyBased(model)
        elif detector_name == "MaxLogit":
            # takes maximum logit over the prediction classes as outlier score
            logit_detectors["MaxLogit"] = MaxLogit(model)
        elif detector_name == "TemperatureScaling":
            # Calibrates
            logit_detectors["TemperatureScaling"] = fit_on_branch_outputs(TemperatureScaling, model)
            
        elif detector_name == "OpenMax":
            # The methods determines a center  for each class in the logits space of a model, and then 
            # creates a statistical model of the distances of correct classified inputs. It uses extreme value 
            # theory to detect outliers by fitting a weibull function to the tail of the distance distribution.We 
            # use the activation of the unknown class as outlier score.
            logit_detectors["OpenMax"] = OpenMax(model, samples=30)


        # BELOW: detectors that do not run on logits
        elif detector_name == "MCD":
            # takes maximum [avg softmax probability] over the prediction classes as outlier score
            feature_detectors["MCD-mean"] = non_verbose_mcd(MCD,model, samples=60,mode = "mean")
            feature_detectors["MCD-var"] = non_verbose_mcd(MCD,model, samples=60,mode = "var")

        elif detector_name == "ODIN":
            feature_detectors["ODIN"] = ODIN(model, eps=0.002)

        elif detector_name == "KLMatching":
            # compares distribution of ID logits with predictions, by estimating a posterior distribution for each cluster
            logit_detectors["KLMatching"] = fit_on_branch_outputs(KLMatching, model)


        elif detector_name == "KNN":
            # Fits a nearest neighbor model to the IN samples an uses the distance from the nearest neighbor as outlier score:
            knn_kwargs = {'metric': 'euclidean'}  # You can adjust this as needed
            # RAW_FEATURE_SPACE(model) is the model > dictates input aggregration
 
            feature_detectors["KNN-raw"] = fit_on_branch_outputs(KNN, RAW_FEATURE_SPACE(model), **knn_kwargs)
            feature_detectors["KNN-att"] = fit_on_branch_outputs(KNN, ATTENDED_FEATURE_SPACE(model), **knn_kwargs)
            feature_detectors["KNN-attinv"] = fit_on_branch_outputs(KNN, INVATTENDED_FEATURE_SPACE(model), **knn_kwargs)
            feature_detectors["KNN-trans"] = fit_on_branch_outputs(KNN, TRANSFORMED_FEATURE_SPACE(model), **knn_kwargs)

        elif detector_name == "RMD":
            # Calculates a cluster center mu for each cluster, and a shared covariance matrix S from the data. 
            # Additionally, it fits a background gaussian with mean mu and covariance matrix S to all of the 
            # features and calculates outlier score of an input as the minimum difference between mahalanobis score for a cluster
            # and the mahalanobis score for the background gaussian.
            feature_detectors["RMD-raw"] =  fit_on_branch_outputs(RMD, RAW_FEATURE_SPACE(model))
            feature_detectors["RMD-att"] =  fit_on_branch_outputs(RMD, ATTENDED_FEATURE_SPACE(model))
            feature_detectors["RMD-attinv"] =  fit_on_branch_outputs(RMD, INVATTENDED_FEATURE_SPACE(model))
            feature_detectors["RMD-attinvmaxnorm"] =  fit_on_branch_outputs(RMD, INVATTENDED_FEATURE_SPACE(model))
            feature_detectors["RMD-trans"] =  fit_on_branch_outputs(RMD, TRANSFORMED_FEATURE_SPACE(model))



        elif detector_name == "Mahalanobis":
            feature_detectors["Mahalanobis-raw"] = fit_on_branch_outputs(Mahalanobis, RAW_FEATURE_SPACE(model))
            feature_detectors["Mahalanobis-att"] = fit_on_branch_outputs(Mahalanobis, ATTENDED_FEATURE_SPACE(model))
            feature_detectors["Mahalanobis-attinv"] = fit_on_branch_outputs(Mahalanobis, INVATTENDED_FEATURE_SPACE(model))
            feature_detectors["Mahalanobis-attinvmaxnorm"] = fit_on_branch_outputs(Mahalanobis, INVATTENDED_FEATURE_SPACE(model))
            feature_detectors["Mahalanobis-trans"] = fit_on_branch_outputs(Mahalanobis, TRANSFORMED_FEATURE_SPACE(model))

        else:
            raise NotImplementedError

    """ For now the feature-based methods perform per whole slide by aggregrating the features. Alternatively we could
    apply the ood detection methods on patch level? 
    Also we want to add MC dropout """


    # testing:
        # elif detector_name == "SHE":
            # head = Head(input_size=1024, num_classes=10).to(device).to(device)
            # feature_detectors["SHE"]  = SHE(ATTENDED_FEATURE_SPACE(model) , head=model)
            # pass
    # feature_detectors["ViM"] = ViM(model._compute_attention, d=64, w=model.fc.weight, b=model.fc.bias)
    # feature_detectors["SHE"] = SHE(lambda x: model._compute_attention, model.classifiers)
    # feature_detectors["DICE"] = DICE(model=model, w=model.classifiers.weight, b=model.classifiers.bias, p=0.65)
    # detectors["MultiMahalanobis"] = MultiMahalanobis(
    #     [model.conv1, model.block1, model.block2, model.block3, nn.Sequential(model.bn1, model.relu)]
    # )
    # detectors["Gram"] = Gram(
    #     num_classes=100,
    #     head=nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), model.fc),
    #     feature_layers=[
    #         model.conv1,
    #         model.block1,
    #         model.block2,
    #         model.block3,
    #         nn.Sequential(model.bn1, model.relu),
    #     ],
    # )


    return logit_detectors, feature_detectors

