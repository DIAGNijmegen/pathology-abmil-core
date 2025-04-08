import argparse
import pdb
import os
import math
import logging
log = logging.getLogger(__name__)
from pathlib import Path

# internal imports
from utils.utils import *


# pytorch imports
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


import pandas as pd
import numpy as np
from tqdm import tqdm

# ood detection imports
from pytorch_ood.utils import  TensorBuffer
from bs4 import BeautifulSoup
import umap

from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
from datetime import datetime

from sklearn.metrics import f1_score #Added to calculate F1-score
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
***REMOVED***


def loader_to_buffer(loader,model):
    buffer  = TensorBuffer()
   
    if len(loader) <= 0:
        buffer = {"embedding": torch.tensor([]), "label": torch.tensor([])}
        return buffer
    
    with torch.no_grad():
        for batch in tqdm(loader,total=len(loader),ncols=50):
            x, y = batch
            x = x.to(device)
            # for CLAM features we have to unsqueeze the first dimension 
            x = x.unsqueeze(0) 
            y = y.to(device)
            z = model(x).cpu()
            z = z.view(z.shape[0], -1)  # flatten
            buffer.append("embedding", z)
            buffer.append("label", y)
    return buffer

def dimred_dict_to_plot(projection_result,labels,opacities,subsets,color_mapping,task,experiment,filenames,label_dict,split=None,extra_label="",output="/data/temporary/ivan/DeepDerma/OOD/results/"):
    opacities = np.array(opacities)
    opacities[[np.isnan(q) for q in opacities]] = 1.0
    if len(np.unique(opacities)) == 1:
        opacity_norm = opacities
        opacity_magnitude = 100 - np.array(opacities)
    else:
        # we inver the opacities because the opacities are actually 
        opacities[opacities<0] = np.max(opacities) # [-1,0.4,..,1] -> [1,0.4,..,1]
        opacities = np.array(opacities)*100 # [0.4,..,1] -> [40,..,100]
        opacity_magnitude = 100 - np.array(opacities) # [40,..,100]  -> [60,..,0] 
        opacity_norm = opacity_magnitude * (1/opacity_magnitude.max()) #  [60,..,0]  -> [1.0,..,0]

    df_reduced = pd.DataFrame({
        "DIM-1": projection_result[:, 0],
        "DIM-2": projection_result[:, 1],
        "PIDs": filenames,
        "Cluster": ["In distribution" if lbl >= 0 else "Out of distribution" for lbl in labels],
        "Subset": subsets,
        "Class": [label_dict[lbl] for lbl in labels],
        "Symbol": ["circle" if lbl >= 0 else "square" for lbl in labels],
        "Size": [6] * len(labels),
        "Opacity": [0.8] * len(labels),
        "InvScore": [f"{f:.2f}" for f in opacity_magnitude],
        "Score": [f"{qs:.2f}" for qs in opacities],
        "OpacityArtifact": [(0.7 * afs)+0.2 for afs in opacity_norm] , 
        "Border": [False if _set == 0 else True for _set in subsets]
    })

    fig = go.Figure()

    for class_name, group in df_reduced.groupby("Class"):
        fig.add_trace(go.Scatter(
            x=group["DIM-1"],
            y=group["DIM-2"],
            mode="markers",
            hovertext = [ f"{p} : AFS {afs}" for (p,afs) in zip(group["PIDs"],group["ArtifactScore"])],
            marker=dict(
                size = list(group["Size"]), 
                opacity=list(group["OpacityArtifact"]), 
                symbol = group["Symbol"],
                line=dict(
                    width=[1]*len(labels),
                    color=["rgba(0,0,0)" if border else color_mapping[class_name] for border in group["Border"]]
                ),
                color=color_mapping[class_name],
            ),
            name=class_name,
            legendgroup=class_name,
            showlegend=False
        ))
        # Dummy trace for legend, to have legend icons without black border
        fig.add_trace(go.Scatter(
            x=[None],  # Empty trace, only for legend
            y=[None],
            mode="markers",
            marker=dict(
                size=8, 
                opacity=0.8,
                symbol = group["Symbol"].iloc[0],
                line=dict(width=2,color=color_mapping[class_name]) ,
                color=color_mapping[class_name],    
            ),
            name=class_name,
            legendgroup=class_name,
            showlegend=True  # Only this trace appears in legend
        ))

    fig.update_layout(
        title=f"Projection. ID data: [{' '.join(task.split('_vs')[0].split('_')[1:])}]| OOD data : [{task.split('_vs_')[1]}] ",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        legend_title="Class"
    )

    fig.show()


    output_html = Path(output) / f"{experiment}/{task}_{split}_{extra_label}.html"
    Path(output_html).parent.mkdir(parents=True, exist_ok=True) 

    fig.write_html(output_html)

    """" Insert HTML code for search functionality"""
    begin_body_file = "./utils/begin_html_insertion.html"
    end_body_file = "./utils/end_html_insertion.html"

    # Read the beginning and ending code blocks
    with open(begin_body_file, "r", encoding="utf-8") as file:
        begin_body_content = file.read()

    with open(end_body_file, "r", encoding="utf-8") as file:
        end_body_content = file.read()

    # Load the main HTML file
    with open(output_html, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Find the body tag
    body_tag = soup.body

    if body_tag:
        # Insert the beginning content at the start of the body
        body_tag.insert(0, BeautifulSoup(begin_body_content, "html.parser"))

        # Append the ending content at the end of the body
        body_tag.append(BeautifulSoup(end_body_content, "html.parser"))

        # Save the modified HTML file
        with open(output_html, "w", encoding="utf-8") as file:
            file.write(str(soup.prettify()))

        print("HTML updated successfully!")

        print(f"Projection saved in: {output_html}")

def dimred_features(loader_A, loader_B, filenames, opacities, model, task, experiment, label_dict=None, split=None, extra_label="", output="/data/temporary/ivan/DeepDerma/OOD/results/"):
    unique_classes = label_dict.values()
    color_palette = pc.qualitative.Light24  # Can be 'Set2', 'Dark24', 'Pastel', etc.
    color_mapping = {cls: color_palette[i % len(color_palette)] for i, cls in enumerate(unique_classes)}
    
    # Load embeddings once
    aggregated_vectors_A, labels_A = load_embeddings(loader_A, model)
    aggregated_vectors_B, labels_B = load_embeddings(loader_B, model)

    labels = np.concatenate((labels_A, labels_B), axis=0).flatten()
    subsets = [0]*len(labels_A) + [1]*len(labels_B)

    # Perform TSNE
    projection_results = features_to_TSNE(aggregated_vectors_A, aggregated_vectors_B)
    dimred_dict_to_plot(projection_results,labels,opacities,subsets, color_mapping, f"TSNE_{task}", experiment,filenames,label_dict, split, extra_label, output)

    # Perform UMAP
    projection_results = features_to_UMAP(aggregated_vectors_A,aggregated_vectors_B)
    dimred_dict_to_plot(projection_results,labels,opacities,subsets, color_mapping, f"UMAP_{task}", experiment,filenames,label_dict, split, extra_label, output)


def load_embeddings(loader, model):
    """
    Function to load embeddings for a given loader (either A or B) and return the embeddings and labels.
    """
    buffer = loader_to_buffer(loader, model)
    z = buffer.get("embedding").detach().to("cpu")
    y = buffer.get("label")
    embeddings = np.array(z)
    labels = np.array(y)
    return embeddings, labels


def features_to_UMAP(aggregated_vectors_A, aggregated_vectors_B):
    print("UMAP: start \n")
    
    # Fit UMAP on loader_A embeddings
    umap_model = umap.UMAP(n_components=2)

    if (aggregated_vectors_B.shape[0] == 0) and (aggregated_vectors_A.shape[0] == 0):   
        raise ValueError("Both loaders have no embeddings")
    
    elif aggregated_vectors_A.shape[0] == 0:
        aggregated_vectors_A = np.reshape(aggregated_vectors_A,(0,aggregated_vectors_B.shape[1]))
        umap_model.fit(aggregated_vectors_B)
    elif aggregated_vectors_B.shape[0] == 0:
        aggregated_vectors_B = np.reshape(aggregated_vectors_B,(0,aggregated_vectors_A.shape[1]))
        umap_model.fit(aggregated_vectors_A)
    else:
        umap_model.fit(aggregated_vectors_A)

    aggregated_vectors = np.concatenate((aggregated_vectors_A, aggregated_vectors_B), axis=0)

    # Transform loader_A and loader_B embeddings using fitted UMAP
    umap_results = umap_model.transform(aggregated_vectors)


    return umap_results


def features_to_TSNE(aggregated_vectors_A, aggregated_vectors_B):
    print("TSNE: start \n")
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    # Concatenate both A and B embeddings
    if (aggregated_vectors_B.shape[0] == 0) and (aggregated_vectors_A.shape[0] == 0):   
        raise ValueError("Both loaders have no embeddings")
    elif aggregated_vectors_A.shape[0] == 0:
        aggregated_vectors_A = np.reshape(aggregated_vectors_A,(0,aggregated_vectors_B.shape[1]))
    elif aggregated_vectors_B.shape[0] == 0:
        aggregated_vectors_B = np.reshape(aggregated_vectors_B,(0,aggregated_vectors_A.shape[1]))

    aggregated_vectors = np.concatenate((aggregated_vectors_A, aggregated_vectors_B), axis=0)
    tsne_results = tsne.fit_transform(aggregated_vectors)
    return tsne_results


