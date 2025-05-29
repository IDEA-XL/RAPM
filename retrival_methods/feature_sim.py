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

# 全局ESM2-650M模型和tokenizer
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
                # 取[CLS] token的特征
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
    
    np.save(f"{task_name}_train_features.npy", train_features)
    np.save(f"{task_name}_test_features.npy", test_features)
    
    
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
    # 构造train/test标签
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

# def RAG_prompt_construction(db_seqs, db_labels, db_features, train_labels, test_insts, test_seqs, test_labels, test_metas, task_name, topk, faiss_index):
    
#     # --- Faiss 检索 ---
#     if faiss_index is None:
#         d = db_features.shape[1]
#         faiss_index = faiss.IndexFlatIP(d)
#         # 归一化特征向量
#         db_features_norm = db_features / np.linalg.norm(db_features, axis=1, keepdims=True)
#         faiss_index.add(db_features_norm.astype(np.float32))
    
#     # 正确加载test的faiss特征
#     test_features = np.load(f"{task_name}_msi_0.0_new_OOD_OOD_test_features.npy")
#     # 归一化查询特征向量
#     test_features_norm = test_features / np.linalg.norm(test_features, axis=1, keepdims=True)
#     st_time = time.time()
#     D, I = faiss_index.search(test_features_norm.astype(np.float32), topk)
#     print(f"Faiss search time: {time.time() - st_time:.4f} seconds")
    
#     faiss_results = []
#     for i, (idxs, scores) in enumerate(zip(I, D)):
#         faiss_topk = []
#         for idx, score in zip(idxs, scores):
#             faiss_topk.append({
#                 # "db_seq": db_seqs[idx],
#                 "db_label": db_labels[idx],
#                 "db_faiss_score": score  # L2距离，取负数作为相似度
#             })
#         faiss_results.append(faiss_topk)

#     print(f"Faiss Top{topk} results for first test sample:")
#     for item in faiss_results[0]:
#         print(item)

#     # --- MMSeqs2 检索 (os.system模拟命令行) ---
#     with tempfile.TemporaryDirectory() as tmpdir:
#         db_fasta = os.path.join(tmpdir, "db.fasta")
#         query_fasta = os.path.join(tmpdir, "query.fasta")
#         result_tsv = os.path.join(tmpdir, "result.tsv")
#         db_dir = os.path.join(tmpdir, "db")
#         query_dir = os.path.join(tmpdir, "query")

#         # 写入fasta
#         with open(db_fasta, "w") as f:
#             for i, seq in enumerate(db_seqs):
#                 f.write(f">db_{i}\n{seq}\n")
#         with open(query_fasta, "w") as f:
#             for i, seq in enumerate(test_seqs):
#                 f.write(f">query_{i}\n{seq}\n")

#         # MMSeqs2 构建数据库
#         os.system(f"mmseqs createdb {db_fasta} {db_dir}")
#         os.system(f"mmseqs createdb {query_fasta} {query_dir}")
#         # MMSeqs2 检索（eary-search）
#         os.system(f"mmseqs easy-search -v 0 -e 1e5 {query_fasta} {db_fasta} {result_tsv} {tmpdir} --max-seqs {topk} --format-output 'query,target,pident'")
        
#         # 解析结果
#         mmseqs_results = [[] for _ in range(len(test_seqs))]
#         with open(result_tsv) as f:
#             for line in f:
#                 qid, tid, pident = line.strip().split('\t')
#                 qidx = int(qid.replace("query_", ""))
#                 tidx = int(tid.replace("db_", ""))
#                 mmseqs_results[qidx].append({
#                     # "db_seq": db_seqs[tidx],
#                     "db_label": db_labels[tidx],
#                     "mmseqs_identity": float(pident)
#                 })

#     print(f"MMSeqs2 Top{topk} results for first test sample:")
#     for item in mmseqs_results[0]:
#         print(item)

#     # --- Faiss 检索（train set self-query，使用 cosine 相似度）---
#     train_features = np.load(f"{task_name}_msi_0.0_new_OOD_OOD_train_features.npy")
#     # 归一化特征向量
#     train_features_norm = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)
#     # 使用内积作为 cosine 相似度
#     train_faiss_index = faiss.IndexFlatIP(train_features_norm.shape[1])
#     train_faiss_index.add(train_features_norm.astype(np.float32))
#     # 查询时也归一化
#     train_D, train_I = train_faiss_index.search(train_features_norm.astype(np.float32), topk)
#     train_faiss_results = []
#     for idxs, scores in zip(train_I, train_D):
#         topk_list = []
#         for idx, score in zip(idxs, scores):
#             topk_list.append({
#                 "train_seqs_label": train_labels[idx],
#                 "train_faiss_score": score  # 这里的score即为cosine相似度
#             })
#         train_faiss_results.append(topk_list)

#     print(f"Faiss Top{topk} results for first train sample:")
#     for item in train_faiss_results[0]:
#         print(item)

#     print("Ground Truth for first test sample:")
#     print("meta: ", test_metas[0])
#     print("label: ", test_labels[0])
    
#     # Construct RAG Prompt in English and save as JSON
#     # The prompt is designed for direct use with GPT, Qwen, Llama3 and similar LLMs.
#     # It instructs the model to only output the answer in the specified JSON format.

#     output_json = []
#     for i in range(len(test_insts)):
#         retrieved_info = []
#         # 添加Faiss检索结果
#         for item in faiss_results[i]:
#             retrieved_info.append({
#                 "db_label": item["db_label"],
#                 "db_faiss_similarity": item["db_faiss_score"]  # negative L2 distance as similarity
#             })
#         # 添加MMSeqs2检索结果
#         mmseqs_info = []
#         for item in mmseqs_results[i]:
#             mmseqs_info.append({
#                 "db_label": item["db_label"],
#                 "mmseqs_identity": item["mmseqs_identity"]
#             })
#         train_examples = []
#         for item in train_faiss_results[i]:
#             train_examples.append({
#                 "example answer": item["train_seqs_label"],
#                 "train_faiss_similarity": item["train_faiss_score"] 
#             })
#         rag_prompt = {
#             "instructions": test_insts[i],
#             "sequence": test_seqs[i],
#             "labels": test_labels[i],
#             "meta_label": test_metas[i],
#             "RAG_prompt": (
#                 f"You are given a protein sequence and two lists of related proteins retrieved from a database.\n"
#                 f"Instruction: {test_insts[i]}\n"
#                 f"Protein sequence: {test_seqs[i]}\n"
#                 f"Retrieved proteins by Faiss (with their labels and similarity scores): {retrieved_info}\n"
#                 f"Retrieved proteins by MMSeqs2 (with their labels and sequence identity): {mmseqs_info}\n"
#                 f"Here are some example input-output pairs for this task:\n"
#                 f"{train_examples}\n"
#                 "Based on the instruction, the protein sequence, the retrieved information, and the examples, "
#                 "output ONLY the functional description of this protein in the following JSON format:\n"
#                 "{\"description\": \"...\"}\n"
#                 "Do not output any other text or explanation. Only output the JSON answer."
#             )
#         }
#         output_json.append(rag_prompt)

#     with open(f"{task_name}_RAP_Top_{topk}.json", "w", encoding="utf-8") as f:
#         json.dump(output_json, f, ensure_ascii=False, indent=2)

#     # # 返回两个检索结果
#     # return faiss_results, mmseqs_results

if __name__ == "__main__":
    
    tasks = ["domain_motif", "catalytic_activity", "general_function", "protein_function"]
    
    all_train_seqs = []
    all_train_labels = []
    all_train_features = []
    
    for now_task in tasks:
        # 读取训练集
        now_train_feature = np.load(f"{now_task}_msi_0.0_new_OOD_OOD_train_features.npy")
        with open(f"/home/wujuntong/MinSimPro/ProteinText/mol-inst-newsplit/{now_task}_msi_0.0_new_OOD_OOD.json", "r", encoding="utf-8") as f:
            dic = json.load(f)
        now_train_seqs = [d["sequence"] for d in dic if d['split'] == 'train']
        now_train_labels = [d["metadata"] for d in dic if d['split'] == 'train']
        all_train_seqs.extend(now_train_seqs)
        all_train_labels.extend(now_train_labels)
        all_train_features.extend(now_train_feature)
        
    all_train_features = np.array(all_train_features)
        
    for top_k in [10]:
        for now_task in tasks:
            print(f"=== Task: {now_task} ===")
            with open(f"/home/wujuntong/MinSimPro/ProteinText/mol-inst-newsplit/{now_task}_msi_0.0_new_OOD_OOD.json", "r", encoding="utf-8") as f:
                dic = json.load(f)
            now_test_instructions = [d["instruction"] for d in dic if d['split'] == 'test']
            now_test_seqs = [d["sequence"] for d in dic if d['split'] == 'test']
            now_test_labels = [d["description"] for d in dic if d['split'] == 'test']
            now_test_meta = [d["metadata"] for d in dic if d['split'] == 'test']
            
            now_train_seqs = [d["sequence"] for d in dic if d['split'] == 'train']
            now_train_labels = [d["description"] for d in dic if d['split'] == 'train']
            
            # RAG_prompt_construction(db_seqs=all_train_seqs,
            #                         db_labels=all_train_labels,
            #                         db_features=all_train_features,
            #                         train_labels=now_train_labels,
            #                         test_insts=now_test_instructions,
            #                         test_seqs=now_test_seqs,
            #                         test_labels=now_test_labels,
            #                         test_metas=now_test_meta,
            #                         task_name=now_task,
            #                         topk=top_k,
            #                         faiss_index=None)
            
            
        
    
    
    # # 示例数据
    # train_seqs = ["MKT...", "ABC..."]
    # test_seqs = ["GHI...", "DEF..."]
    # train_labels = [1, 0]
    # test_labels = [0, 1]
    # task_name = "example_task"
    
    # peer_knn_predict(train_seqs, test_seqs, train_labels, test_labels, task_name)