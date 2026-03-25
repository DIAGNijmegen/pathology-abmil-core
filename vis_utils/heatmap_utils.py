import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import pandas as pd
from utils.utils import *
from PIL import Image
from math import floor
import matplotlib.pyplot as plt
from dataset_modules.wsi_dataset import Wsi_Region
from utils.transform_utils import get_eval_transforms
import h5py
from wsi_core.WholeSlideImage import WholeSlideImage
from scipy.stats import percentileofscore
import math
from utils.file_utils import save_hdf5
from scipy.stats import percentileofscore
from utils.constants import MODEL2CONSTANTS
from tqdm import tqdm

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def score2percentile(score, ref): #https://github.com/mahmoodlab/CLAM/issues/153
    percentile = percentileofscore(ref, score)
    return percentile

def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level = -1, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)
    
    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)
    
    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap

def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
    wsi_object = WholeSlideImage(wsi_path)
    if seg_params['seg_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        seg_params['seg_level'] = best_level

    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
    wsi_object.saveSegmentation(seg_mask_path)
    return wsi_object
#Compute bag_logits, patch_logits from batches 
def compute_from_patches(wsi_object, img_transforms, feature_extractor=None, clam_pred=None, model=None, model_type=None, batch_size=512,  
    attn_save_path=None, ref_scores=None, feat_save_path=None, **wsi_kwargs):    
    top_left = wsi_kwargs['top_left']
    bot_right = wsi_kwargs['bot_right']
    patch_size = wsi_kwargs['patch_size'] 
    
    roi_dataset = Wsi_Region(wsi_object, t=img_transforms, **wsi_kwargs)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=0)
    print('total number of patches to process: ', len(roi_dataset))
    num_batches = len(roi_loader)
    print('number of batches: ', num_batches)
    mode = "w"
    for idx, (roi, coords) in enumerate(tqdm(roi_loader)):
        roi = roi.to(device)
        coords = coords.numpy()
        
        with torch.inference_mode():
            features = feature_extractor(roi)

            if attn_save_path is not None:
                if (model_type == 'addmil') and (idx < len(roi_loader)-1):
                    # Save features in a temporary h5 file until all the features are extracted
                    feat_prod_save_path = attn_save_path.replace('.h5', '_feat_prod.h5')
                    d_feat_prod = {'features': features.cpu().numpy(), 'coords': coords}    
                    save_hdf5(feat_prod_save_path, d_feat_prod, mode=mode)
                elif (model_type == 'addmil') and (idx == len(roi_loader)-1):
                    feat_prod_save_path = attn_save_path.replace('.h5', '_feat_prod.h5')
                    d_feat_prod = {'features': features.cpu().numpy(), 'coords': coords}    
                    save_hdf5(feat_prod_save_path, d_feat_prod, mode=mode)

                    # Load the features from the temporary h5 file
                    feat_prod_h5 = h5py.File(feat_prod_save_path, mode='r')
                    features_all = torch.tensor(feat_prod_h5['features']).to(device)
                    coords_all = feat_prod_h5['coords'][:]
                    feat_prod_h5.close()
                    os.remove(feat_prod_save_path)

                    # Compute the attention scores 
                    logits, att_raw, results_dict = model(features_all) 
                    patch_logits = results_dict['patch_logits'].cpu().numpy()

                    # Squeeze A as a placeholder
                    A = results_dict['patch_logits'][...,clam_pred].view(-1, 1).cpu().numpy()

                    asset_dict = {'patch_logits': patch_logits, 'attention_scores': A, 'coords': coords_all}
                    save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)
                else: #CLAM original handling
                    A = model(features, attention_only=True)
           
                    if A.size(0) > 1: #CLAM multi-branch attention
                        A = A[clam_pred]

                    A = A.view(-1, 1).cpu().numpy()

                    if ref_scores is not None:
                        for score_idx in range(len(A)):
                            A[score_idx] = score2percentile(A[score_idx], ref_scores)

                    asset_dict = {'attention_scores': A, 'coords': coords}
                    save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)
    
        if feat_save_path is not None:
            asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
            save_hdf5(feat_save_path, asset_dict, mode=mode)

        mode = "a"
        
    #Weighting and softmax the patch_logits by size of the image and softmax the whole image logits 
    if (attn_save_path is not None and model_type == 'addmil'):
        #open a hdf5 file and read the attention scores
        attn_h5_temp = h5py.File(attn_save_path, mode='r+')         ## open the file
        patch_logits = attn_h5_temp['patch_logits']                 ## load the data 
        attention_scores = attn_h5_temp['attention_scores']         ## load the attention scores
        patch_logits[...] = patch_logits[...] * patch_logits[...].shape[0]            ## weighting the patch_logits by size of the image (number of patches)
        attention_scores_unreshaped = F.softmax(torch.tensor(patch_logits[...]), dim=1).cpu().numpy()[...,clam_pred]
        attention_scores[...] = attention_scores_unreshaped.reshape(attention_scores_unreshaped.shape[0], 1) ## softmax the whole image logits
        ##sumcheck_weighted = A[:,:,0] + A[:,:,1] #sumcheck_weighted = 1
        attn_h5_temp.close()                                        ## close the file

    return attn_save_path, feat_save_path, wsi_object
