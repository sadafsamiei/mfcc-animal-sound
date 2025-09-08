import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.MFCC import MFCCDataset, get_dataloaders, collate_fn
from assets import RESULTS_DIR

def compute_class_centroids(train_set):
    centroids = {}
    for x, y in train_set:
        y = y.item()
        vec = x.mean(dim=0).numpy()  
        if y not in centroids:
            centroids[y] = []
        centroids[y].append(vec)

    for k in centroids:
        centroids[k] = np.mean(np.stack(centroids[k], axis=0), axis=0)
    return centroids

def plot_distance_hist(threshold=175.0, batch_size=1):
    train_loader, _, label2idx, idx2label = get_dataloaders(batch_size=batch_size)
    train_set = train_loader.dataset
    val_set = MFCCDataset("val", label2idx=label2idx) 
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    centroids = compute_class_centroids(train_set)

    dists_id, dists_near, dists_far = [], [], []

    for x, y in val_loader:
        x = x[0]  
        vec = x.mean(dim=0).numpy()

        if y.item() == -1:  
            min_dist = min(np.linalg.norm(vec - c) for c in centroids.values())
            if min_dist < threshold:
                dists_near.append(min_dist) 
            else:
                dists_far.append(min_dist)   
        else:  
            centroid = centroids[y.item()]
            dist = np.linalg.norm(vec - centroid)
            dists_id.append(dist)

    plt.figure(figsize=(8, 6))
    bins = 20

    if dists_id:
        plt.hist(dists_id, bins=bins, alpha=0.6, color="green", label="In-Distribution")
    if dists_near:
        plt.hist(dists_near, bins=bins, alpha=0.6, color="orange", label="Near-OOD")
    if dists_far:
        plt.hist(dists_far, bins=bins, alpha=0.6, color="red", label="Far-OOD")

    plt.axvline(threshold, color="black", linestyle="--", linewidth=2, label="Threshold")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.title("Distance to Class Centroids")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "centroid_distance.png")
    plt.savefig(out_path)
    plt.close()

    print("\nCentroid Distance Summary")
    print("=" * 40)
    print(f"In-Distribution samples : {len(dists_id)}")
    print(f"Near-OOD samples        : {len(dists_near)}")
    print(f"Far-OOD samples         : {len(dists_far)}")
    print(f"Threshold set at        : {threshold}")
    print(f"Plot saved to: {out_path}")

if __name__ == "__main__":
    plot_distance_hist(threshold=100.0)
