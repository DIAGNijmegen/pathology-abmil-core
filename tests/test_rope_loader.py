"""Test script for RoPE, loaders, collate, and AttentionSingleBranch forward.

Run with:

python -m tests.test_rope_loader

This script creates a small dummy dataset with one slide (batch_size=1), tests
`collate_MIL_rope`, `get_simple_loader(..., use_rope=True)`, and runs a forward
pass through `AttentionSingleBranch(use_rope=True)` using the produced coords.
"""
import os
import sys
import numpy as np
import torch

# ensure local package imports work
HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(HERE, '..'))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from utils.utils import collate_MIL_rope, get_simple_loader
from models.attentionhead import AttentionSingleBranch


def make_dummy_slide(n_patches=10, feat_dim=1024):
    feats = torch.randn(n_patches, feat_dim).float()
    coords = (np.random.rand(n_patches, 2) * 1000.0).astype(np.float32)
    return feats, coords


class DummyMILDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        # items: list of tuples (features:torch.Tensor [N,C], label:int, coords: np.ndarray [N,2])
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        feats, label, coords = self.items[idx]
        return feats, label, coords


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # create a single-slide dataset (batch_size is expected to be 1 for these collates)
    feat_dim = 1024
    n_patches = 7
    feats, coords = make_dummy_slide(n_patches=n_patches, feat_dim=feat_dim)
    items = [(feats, 1, coords)]
    ds = DummyMILDataset(items)

    # test collate directly
    batch = [ds[0]]
    img, label, coords_coll = collate_MIL_rope(batch)
    print("collate_MIL_rope -> img.shape:", img.shape)
    print("collate_MIL_rope -> label.shape:", label.shape)
    print("collate_MIL_rope -> coords.shape:", coords_coll.shape)

    assert img.shape == (n_patches, feat_dim)
    assert label.shape == (1,) or label.shape[0] == 1
    assert coords_coll.shape == (1, n_patches, 2)

    # test DataLoader wrapper
    loader = get_simple_loader(ds, batch_size=1, num_workers=0, use_rope=True)
    for batch_data in loader:
        img2, label2, coords2 = batch_data
        print("Loader returned img2.shape:", img2.shape)
        print("Loader returned label2.shape:", label2.shape)
        print("Loader returned coords2.shape:", coords2.shape)
        break

    # prepare inputs for AttentionSingleBranch: 
    # x can be [N, C] or [B, N, C], _apply_rope handles both
    # coords should be [B, N, 2]
    x = img2  # [N, C] - no unsqueeze needed, _apply_rope handles it
    coords_t = coords2  # [B, N, 2]

    # instantiate model and run forward
    model = AttentionSingleBranch(use_rope=True)
    model.relocate()  # move rope_proj to correct device
    model.eval()

    # convert coords to float tensor on same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    coords_t = coords_t.to(device)
    model = model.to(device)

    with torch.no_grad():
        out = model(x, coords=coords_t)

    print("Model forward returned:")
    if isinstance(out, tuple) or isinstance(out, list):
        for i, o in enumerate(out):
            if isinstance(o, dict):
                print(f" - output[{i}] is dict with keys: {list(o.keys())}")
            elif isinstance(o, torch.Tensor):
                print(f" - output[{i}].shape = {o.shape}")
            else:
                print(f" - output[{i}] type {type(o)}")
    else:
        print(type(out))

    # test _apply_rope produces a different tensor (rotation applied)
    model_cpu = model.to('cpu')
    x_cpu = x.to('cpu')
    coords_cpu = coords_t.to('cpu')
    applied = model_cpu._apply_rope(x_cpu, coords_cpu)
    if torch.allclose(applied, x_cpu):
        print("Warning: _apply_rope produced identical tensor to input (unexpected)")
    else:
        print("_apply_rope produced modified features (as expected)")

    print("All tests in test_rope_loader completed successfully.")


if __name__ == '__main__':
    main()
