import os
import json
import time
import tempfile
import numpy as np
import faiss
import sys
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel


def extract_features(dataset, split_name, feature_dir="features"):
    ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)
    model = AutoModel.from_pretrained(ESM2_MODEL_NAME).to(DEVICE)
    model.eval()
    
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


def score_to_confidence(score):
    # 将分数转为置信度，大于90为High Confindence, 60到90为Medium Confidence, 60以下为Low Confidence
    if score >= 90:
        return "<High Confidence>"
    elif score >= 60:
        return "<Medium Confidence>"
    else:
        return "<Low Confidence>"
    

def RAG_prompt_construction(db_seqs, db_labels, db_features, train_labels, test_insts, test_seqs, test_labels, test_metas, task_name, topk, faiss_index):
    
    # --- Faiss 检索 (HNSW Index) ---
    if faiss_index is None:
        d = db_features.shape[1]
        faiss_index = faiss.IndexHNSWFlat(d, 32)
        db_features_norm = db_features / np.linalg.norm(db_features, axis=1, keepdims=True)
        faiss_index.add(db_features_norm.astype(np.float32))
        faiss_index.hnsw.efSearch = max(50, topk * 2)
    
    test_features = np.load(f"{task_name}_test_features.npy")
    test_features_norm = test_features / np.linalg.norm(test_features, axis=1, keepdims=True)
    st_time = time.time()
    D, I = faiss_index.search(test_features_norm.astype(np.float32), topk)
    print(f"Faiss HNSW search time: {time.time() - st_time:.4f} seconds")
    
    faiss_results = []
    for i, (idxs, scores) in enumerate(zip(I, D)):
        faiss_topk = []
        for idx, score in zip(idxs, scores):
            faiss_topk.append({
                # "db_seq": db_seqs[idx],
                "db_label": db_labels[idx],
                "score": score 
            })
        faiss_results.append(faiss_topk)

    # print(f"Faiss Top{topk} results for first test sample:")
    # for item in faiss_results[0]:
    #     print(item)

    # --- MMSeqs2 检索 (os.system模拟命令行) ---
    with tempfile.TemporaryDirectory() as tmpdir:
        db_fasta = os.path.join(tmpdir, "db.fasta")
        query_fasta = os.path.join(tmpdir, "query.fasta")
        result_tsv = os.path.join(tmpdir, "result.tsv")
        db_dir = os.path.join(tmpdir, "db")
        query_dir = os.path.join(tmpdir, "query")

        with open(db_fasta, "w") as f:
            for i, seq in enumerate(db_seqs):
                f.write(f">db_{i}\n{seq}\n")
        with open(query_fasta, "w") as f:
            for i, seq in enumerate(test_seqs):
                f.write(f">query_{i}\n{seq}\n")

        os.system(f"mmseqs createdb {db_fasta} {db_dir}")
        os.system(f"mmseqs createdb {query_fasta} {query_dir}")
        os.system(f"mmseqs easy-search -v 0 -e 1e5 {query_fasta} {db_fasta} {result_tsv} {tmpdir} --max-seqs {topk} --format-output 'query,target,pident'")
        
        mmseqs_results = [[] for _ in range(len(test_seqs))]
        with open(result_tsv) as f:
            for line in f:
                qid, tid, pident = line.strip().split('\t')
                qidx = int(qid.replace("query_", ""))
                tidx = int(tid.replace("db_", ""))
                mmseqs_results[qidx].append({
                    "db_label": db_labels[tidx],
                    "score": float(pident)
                })

    train_features = np.load(f"{task_name}_train_features.npy")
    train_features_norm = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)
    train_faiss_index = faiss.IndexHNSWFlat(train_features_norm.shape[1], 32)
    train_faiss_index.hnsw.efSearch = max(50, topk * 2)
    train_faiss_index.add(train_features_norm.astype(np.float32))
    train_D, train_I = train_faiss_index.search(train_features_norm.astype(np.float32), topk)
    train_faiss_results = []
    for idxs, scores in zip(train_I, train_D):
        topk_list = []
        for idx, score in zip(idxs, scores):
            topk_list.append({
                "train_seqs_label": train_labels[idx],
                "confidence_level": score_to_confidence(score * 100),
            })
        train_faiss_results.append(topk_list)
    
    output_json = []
    for i in range(len(test_insts)):
        combined_results = []
        # Add Faiss results
        for item in faiss_results[i]:
            combined_results.append({
            "db_label": item["db_label"],
            "score": item["score"],
            })
        # Add MMSeqs2 results
        for item in mmseqs_results[i]:
            combined_results.append({
            "db_label": item["db_label"],
            "score": item["score"],
            })
        # Sort by score descending, take topk
        combined_results_sorted = sorted(combined_results, key=lambda x: x["score"], reverse=True)[:topk]
        retrieved_info = []
        for item in combined_results_sorted:
            retrieved_info.append({
            "db_label": item["db_label"],
            "confidence level": score_to_confidence(item["score"]),
            })
        train_examples = []
        for item in train_faiss_results[i]:
            train_examples.append({
                "example answer": item["train_seqs_label"],
                "confidence level": item["confidence_level"] 
            })
        rag_prompt = {
            "instructions": test_insts[i],
            "sequence": test_seqs[i],
            "labels": test_labels[i],
            "meta_label": test_metas[i],
            "RAG_prompt": (
                f"You are given a protein sequence and two lists of related proteins retrieved from a database.\n"
                f"Instruction: {test_insts[i]}\n"
                f"Protein sequence: {test_seqs[i]}\n"
                f"Retrieved proteins annotations by Faiss and MMSeqs2: {retrieved_info}\n"
                f"Here are some example input-output pairs for this task:\n"
                f"{train_examples}\n"
                "Based on the instruction, the protein sequence, the retrieved information, and the examples, "
                "output ONLY the functional description of this protein in the following JSON format:\n"
                "{\"description\": \"...\"}\n"
                "Do not output any other text or explanation. Only output the JSON answer."
            )
        }
        output_json.append(rag_prompt)

    with open(f"{task_name}_RAP_Top_{topk}.json", "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print("Usage: python RAG_prompt_cons.py <dataset_path> <top_k>")
        sys.exit(1)
    dataset_path = sys.argv[1]
    top_k = int(sys.argv[2])
    
    all_train_seqs = []
    all_train_labels = []
    all_train_features = []
    
    
    # for now_task in tasks:
    for p in os.listdir(dataset_path):
        now_task = p[:-5]
        print(f"=== Task: {now_task} ===")
        if not p.endswith(".json"): continue
        JSON_PATH = os.path.join(dataset_path, p)
        dic = json.load(open(JSON_PATH, "r"))
        
        try:
            now_train_feature = np.load(f"{now_task}_train_features.npy")
        except FileNotFoundError:
            print(f"Feature file for {now_task} not found, extracting features...")
            
            train_dic = [d for d in dic if d["split"] == "train"]
            test_dic = [d for d in dic if d["split"] == "test"]
            train_seqs = [d["sequence"] for d in train_dic]
            train_labels = [d["description"] for d in train_dic]
            test_seqs = [d["sequence"] for d in test_dic]
            test_labels = [d["description"] for d in test_dic]
            train_set = [{"seq": seq, "label": label} for seq, label in zip(train_seqs, train_labels)]
            test_set = [{"seq": seq, "label": label} for seq, label in zip(test_seqs, test_labels)]
            feature_dir = f"features/{now_task}"
            now_train_feature, _ = extract_features(train_set, "train", feature_dir)
            now_test_features, _ = extract_features(test_set, "test", feature_dir)
            np.save(f"{now_task}_train_features.npy", now_train_feature)
            np.save(f"{now_task}_test_features.npy", now_test_features)
            
        now_train_seqs = [d["sequence"] for d in dic if d['split'] == 'train']
        now_train_labels = [d["metadata"] for d in dic if d['split'] == 'train']
        all_train_seqs.extend(now_train_seqs)
        all_train_labels.extend(now_train_labels)
        all_train_features.extend(now_train_feature)
        
    all_train_features = np.array(all_train_features)
        
    for p in os.listdir(dataset_path):
        now_task = p[:-5]
        print(f"=== Task: {now_task} ===")
        JSON_PATH = os.path.join(dataset_path, p)
        dic = json.load(open(JSON_PATH, "r"))
        now_test_instructions = [d["instruction"] for d in dic if d['split'] == 'test']
        now_test_seqs = [d["sequence"] for d in dic if d['split'] == 'test']
        now_test_labels = [d["description"] for d in dic if d['split'] == 'test']
        now_test_meta = [d["metadata"] for d in dic if d['split'] == 'test']
        
        now_train_seqs = [d["sequence"] for d in dic if d['split'] == 'train']
        now_train_labels = [d["description"] for d in dic if d['split'] == 'train']
        
        RAG_prompt_construction(db_seqs=all_train_seqs,
                                db_labels=all_train_labels,
                                db_features=all_train_features,
                                train_labels=now_train_labels,
                                test_insts=now_test_instructions,
                                test_seqs=now_test_seqs,
                                test_labels=now_test_labels,
                                test_metas=now_test_meta,
                                task_name=now_task,
                                topk=top_k,
                                faiss_index=None)