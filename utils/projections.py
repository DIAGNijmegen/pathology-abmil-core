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


import anndata
import scanpy as sc
 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
***REMOVED***


def loader_to_buffer(loader,model):
    print("loading embeddings in buffers")
    buffer  = TensorBuffer()
   
    if not loader:
        loader = []
    
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


def category_to_border(plot_data,args):
    if not args.border_tag:
        return [False]*len(plot_data["label"])
    else:
        category_labels = plot_data[args.border_tag]
        unique_categories = np.unique(category_labels)
        if len(unique_categories)> 2:
            raise ValueError("Can not binarize category labels for border")
        else:
            return [True if lbl == unique_categories[0] else False for lbl in category_labels]  


def category_to_symbol(category_labels,binarize=False):
    symbol_map = {
        0: "circle",
        1: "star",
        2: "square",
        3: "diamond",
        4: "star-triangle-down",
        5: "cross",
        6: "x-thin",
        7: "astersk",
        8: "y-up",
        9: "diamond-wide"
    }
    print("n categories for symbols: ", len(set(category_labels)))
    value_to_index = {val: idx for idx, val in enumerate((set(category_labels))) }
    normalized_categories = list(map(value_to_index.get,category_labels))
    return list(map(symbol_map.get,normalized_categories))


def category_to_colors(category_labels,binarize=False):
    color_palette = pc.qualitative.Light24 
    color_mapping = {cls: color_palette[i % len(color_palette)] for i, cls in enumerate(set(category_labels))}
    color_names = list(map(color_mapping.get,category_labels))
    return color_names


def category_to_opacities(args,plot_data,n_items):
    if args.opacity_label:
        opacity_data = plot_data[args.opacity_label]
    else:
        opacity_data = []
    if not args.opacity_label or len(np.unique(opacity_data)) == 1:
        # we don't use opacity 
        opacities = opacity_inv_magnitude= np.array([1.0]*n_items)
        opacity_inv_norm = opacity_norm = np.array([1.0]*n_items)

    else:
        opacities = np.array(opacities)
        opacities[[np.isnan(q) for q in opacities]] = 1.0
        # we inverse the opacities because the opacities are actually 
        opacities[opacities<0] = np.max(opacities) # [-1,0.4,..,1] -> [1,0.4,..,1]
        opacity_norm = opacities * (1/opacity_norm.max())

        opacities = np.array(opacities)*100 # [0.4,..,1] -> [40,..,100] # refers to quality score
        opacity_inv_magnitude = 100 - np.array(opacities) # [40,..,100]  -> [60,..,0] 
        opacity_inv_norm = opacity_inv_magnitude * (1/opacity_inv_magnitude.max()) #  [60,..,0]  -> [1.0,..,0]

    return opacities, opacity_inv_norm, opacity_norm


def plot_projection_data(projection_result,plot_data,task,args,split=None,extra_label=""):
    experiment = args.exp_code
    output = args.projections_results_dir
        
    label_classes = plot_data[args.class_tag]
    color_names = category_to_colors(plot_data[args.color_tag])
    symbols = category_to_symbol(plot_data[args.symbol_tag])
    borders = category_to_border(plot_data,args)
    opacities, opacity_inv_norm, opacity_norm = category_to_opacities(args,plot_data,len(plot_data["label"]))

    df_reduced = pd.DataFrame({
        "DIM-1": projection_result[:, 0],
        "DIM-2": projection_result[:, 1],
        "PIDs": plot_data["slide_id"],
        "Class": label_classes,
        "MetadataString" : [
             " | ".join(f"{key} : {os.path.basename(str(plot_data[key][i]))}" for key in plot_data.keys())
            for i in range(len(plot_data["slide_id"]))
        ],
        "Color" : color_names,
        "Symbol": symbols,
        "Size": [6] * len(label_classes),
        "InvScore": [f"{f:.2f}" for f in opacity_inv_norm],
        "Score": [f"{qs:.2f}" if args.opacity_label  else "None" for qs in opacities ],
        "OpacityArtifact": [(0.7 * afs)+0.2 for afs in opacity_inv_norm] , 
        "OpacityQuality": [(0.7 * afs)+0.2 for afs in opacity_norm] , 
        "Border": borders
    })



    fig = go.Figure()

    for class_name, group in df_reduced.groupby("Class"):
        fig.add_trace(go.Scatter(
            x=group["DIM-1"],
            y=group["DIM-2"],
            mode="markers",
            hovertext=group["MetadataString"],
            marker=dict(
                size = list(group["Size"]), 
                opacity=list(group["OpacityArtifact"]), 
                symbol = group["Symbol"],
                line=dict(
                    width=[1]*len(group["Size"]),
                    color=["rgba(0,0,0)" if border else fill_color for (fill_color,border) in zip(group["Color"],group["Border"])]
                ),
                color=group["Color"],
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
                symbol = group["Symbol"].iloc[0] if (args.class_tag == args.symbol_tag) else "circle" ,
                line=dict(width=2,
                          color = ["rgba(0,0,0)" if ( (args.class_tag == args.border_tag) and border) else fill_color for (fill_color,border) in zip(group["Color"],group["Border"])]
                ),
                color= group["Color"] if (args.class_tag == args.color_tag) else "rgba(169, 169, 169)",
 
            ),
            name=class_name,
            legendgroup=class_name,
            showlegend=True  # Only this trace appears in legend
        ))

    fig.update_layout(
        # title=f"Projection. ID data: [{' '.join(task.split('_vs')[0].split('_')[1:])}]| OOD data : [{task.split('_vs_')[1]}] ",
        title=f"Projection. {task} \n Symbol: {args.symbol_tag} | Color: {args.color_tag} | Border: {args.border_tag} | Opacity: {args.opacity_label}",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        legend_title=f"{args.class_tag.split('-bin')[0]}"
    )

    fig.show()

    fig_to_html(output,experiment,task,split,extra_label,fig)


def fig_to_html(output,experiment,task,split,extra_label,fig):
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


def merge_loader_attributes(_content_items):
    _result = []
    for _content in _content_items:
        try:
            if len(_content) > 0:
                _result.extend(_content)
        except:
            pass
    return _result

def get_default_plot_data(loader_A,loader_B,aggregated_vectors_A,aggregated_vectors_B,label_dict,args):
    print("getting default plot data")
    plot_data = {}
    plot_data["split"] = ["train"]*len(loader_A) + ["val"]*len(loader_B)

    for plot_tag in ["slide_id","label"]:
        print("loading data from : ", plot_tag)
        a_tag = list(loader_A.dataset.slide_data[plot_tag][list(loader_A.sampler)])
        b_tag = list(loader_B.dataset.slide_data[plot_tag][list(loader_B.sampler)])
        plot_data[plot_tag] = merge_loader_attributes([a_tag,b_tag])

    plot_data["distribution"] = ["IN DISTRIBUTION" if lbl >= 0 else "OUT DISTRIBUTION" for lbl in plot_data["label"]] 

    plot_data["cluster"] = features_to_clusters(aggregrate_vectors(aggregated_vectors_A,aggregated_vectors_B))
    plot_data["label"] = [label_dict.get(lbl) for lbl in plot_data["label"]]
    return plot_data    

def extend_default_plot_data(plot_data,loader_A,loader_B,args):
    print("extending default plot data")
    for plot_tag in set([args.symbol_tag,args.class_tag,args.color_tag,args.border_tag,args.opacity_label,"slide_id","label"]):
        if not plot_tag in ["distribution","cluster","split",None]:
            print("loading data from : ", plot_tag)
            a_tag = list(loader_A.dataset.slide_data[plot_tag][list(loader_A.sampler)])
            b_tag = list(loader_B.dataset.slide_data[plot_tag][list(loader_B.sampler)])
            plot_data[plot_tag] = merge_loader_attributes([a_tag,b_tag])
    return plot_data


def make_projections(loader_A=[], loader_B=[], model=None, task=None, label_dict=None, args =None, split=None, extra_label="",make_plot=True):
    if not args.class_tag in [args.color_tag,args.symbol_tag]:
        raise ValueError(f"Invalid color or symbol tag. One of them should be class tag: {args.class_tag}")
    # Load embeddings once
    aggregated_vectors_A, labels_A = load_embeddings(loader_A, model)
    aggregated_vectors_B, labels_B = load_embeddings(loader_B, model)

    # load contents used for plotting
    plot_data = get_default_plot_data(loader_A,loader_B,aggregated_vectors_A,aggregated_vectors_B,label_dict,args)
    plot_data = extend_default_plot_data(plot_data,loader_A,loader_B,args)

    # Perform TSNE
    tsne_results = features_to_TSNE(aggregated_vectors_A, aggregated_vectors_B)
    if make_plot:
        plot_projection_data(projection_result= tsne_results,plot_data=plot_data,args=args,task= f"TSNE_{task}",split=split, extra_label=extra_label)

    # Perform UMAP
    umap_results = features_to_UMAP(aggregated_vectors_A,aggregated_vectors_B)
    if make_plot:
        plot_projection_data(projection_result= umap_results,plot_data=plot_data,args=args,task= f"UMAP_{task}", split=split, extra_label=extra_label)

    if not make_plot:
        return {
            "plot_data": plot_data,
            "tsne": tsne_results,
            "umap": umap_results,
            "args": args,
            "split": split,
            "extra_label": extra_label,
        } 

def features_to_clusters(features):
    print("Performing leiden clutering")
    adata = anndata.AnnData(X=features)  # use full features or PCA output here
    # Build k-NN graph on the t-SNE coordinates
    sc.pp.neighbors(adata, use_rep='X')  # X is the default matrix here (2D t-SNE)
    sc.tl.leiden(adata)

    # View clusters
    clusters = adata.obs["leiden"].values

    return clusters


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


def aggregrate_vectors(aggregated_vectors_A, aggregated_vectors_B):
    if (aggregated_vectors_B.shape[0] == 0) and (aggregated_vectors_A.shape[0] == 0):   
        raise ValueError("Both loaders have no embeddings")
    elif aggregated_vectors_A.shape[0] == 0:
        aggregated_vectors_A = np.reshape(aggregated_vectors_A,(0,aggregated_vectors_B.shape[1]))
    elif aggregated_vectors_B.shape[0] == 0:
        aggregated_vectors_B = np.reshape(aggregated_vectors_B,(0,aggregated_vectors_A.shape[1]))


    aggregated_vectors = np.concatenate((aggregated_vectors_A, aggregated_vectors_B), axis=0)
    return aggregated_vectors


def features_to_UMAP(aggregated_vectors_A, aggregated_vectors_B):
    print("UMAP: start \n")
    
    # Fit UMAP on loader_A embeddings
    umap_model = umap.UMAP(n_components=2)

    if (aggregated_vectors_B.shape[0] == 0) and (aggregated_vectors_A.shape[0] == 0):   
        raise ValueError("Both loaders have no embeddings")
    elif aggregated_vectors_A.shape[0] == 0:
        umap_model.fit(aggregated_vectors_B)
    else:
        umap_model.fit(aggregated_vectors_A)

    aggregated_vectors=aggregrate_vectors(aggregated_vectors_A,aggregated_vectors_B)

    # Transform loader_A and loader_B embeddings using fitted UMAP
    umap_results = umap_model.transform(aggregated_vectors)


    return umap_results


def features_to_TSNE(aggregated_vectors_A, aggregated_vectors_B):
    print("TSNE: start \n")
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    # Concatenate both A and B embeddings
    aggregated_vectors=aggregrate_vectors(aggregated_vectors_A,aggregated_vectors_B)

    tsne_results = tsne.fit_transform(aggregated_vectors)
    return tsne_results


