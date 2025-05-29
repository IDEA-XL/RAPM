import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import faiss

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import subprocess
import tempfile
import json
import time

ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)
model = AutoModel.from_pretrained(ESM2_MODEL_NAME).to(DEVICE)
model.eval()

def extract_features(dataset, split_name, feature_dir="features"):
    os.makedirs(feature_dir, exist_ok=True)
    features = []
    labels = []
    for idx, item in enumerate(tqdm(dataset, desc=f"Extracting {split_name} features")):
        seq = item['seq']
        label = item['label']
        feature_path = os.path.join(feature_dir, f"{split_name}_{idx}.npy")
        if os.path.exists(feature_path):
        # if False:
            feat = np.load(feature_path)
        else:
            # Tokenize and move to device
            inputs = tokenizer(seq, return_tensors="pt", truncation=True, padding=True, max_length=1024)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                feat = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
            np.save(feature_path, feat)
        features.append(feat)
        labels.append(label)
    features = np.stack(features)
    
    return features, labels

def knn_predict(train_features, train_labels, test_features, k=1):
    d = train_features.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(train_features.astype(np.float32))
    D, I = index.search(test_features.astype(np.float32), k)
    # 取最近的一个样本的label
    pred_labels = [train_labels[i[0]] for i in I]
    return pred_labels

def evaluate_accuracy(pred_labels, true_labels):
    correct = sum(p == t for p, t in zip(pred_labels, true_labels))
    return correct / len(true_labels)

def peer_knn_predict(train_seqs, test_seqs, train_labels, test_labels, task_name):
    train_set = [{"seq": seq, "label": label} for seq, label in zip(train_seqs, train_labels)]
    test_set = [{"seq": seq, "label": label} for seq, label in zip(test_seqs, test_labels)]
    feature_dir = f"features/{task_name}"
    os.makedirs(feature_dir, exist_ok=True)
    
    train_features, train_labels = extract_features(train_set, "train", feature_dir)
    test_features, test_labels = extract_features(test_set, "test", feature_dir)
    
    np.save(f"features/{task_name}_train_features.npy", train_features)
    np.save(f"features/{task_name}_test_features.npy", test_features)
    
    pred_labels = knn_predict(train_features, train_labels, test_features, k=1)
    acc = evaluate_accuracy(pred_labels, test_labels)
    print(f"KNN prediction accuracy: {acc:.4f}")
    return pred_labels, test_labels, acc


def tsne_visualization_train_test(features, train_size, task_name):
    
    # tsne_path = f"tsne_{task_name}.npy"
    # if os.path.exists(tsne_path):
    #     reduced_features = np.load(tsne_path)
    # else:
    tsne = TSNE(n_components=2, random_state=42, n_iter_without_progress=1000)
    reduced_features = tsne.fit_transform(features)
        # np.save(tsne_path, reduced_features)
    
    plt.figure(figsize=(10, 8))

    split_labels = ["train"] * train_size + ["test"] * (features.shape[0] - train_size)
    palette = {"train": "blue", "test": "orange"}
    sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=split_labels, palette=palette, sizes=0.3)
    plt.title(f"t-SNE: Train vs Test Distribution for {task_name}")
    plt.savefig(f"tsne_train_test_{task_name}.png")
    plt.show()
    plt.close()

def tsne_analysis_train_test(train_seqs, train_labels, test_seqs, test_labels, task_name):
    train_set = [{"seq": seq, "label": label} for seq, label in zip(train_seqs, train_labels)]
    test_set = [{"seq": seq, "label": label} for seq, label in zip(test_seqs, test_labels)]
    
    feature_dir = f"features/{task_name}"
    os.makedirs(feature_dir, exist_ok=True)
    
    train_features, _ = extract_features(train_set, "train", feature_dir)
    test_features, _ = extract_features(test_set, "test", feature_dir)
    
    all_features = np.concatenate((train_features, test_features), axis=0)
    tsne_visualization_train_test(all_features, len(train_features), task_name)