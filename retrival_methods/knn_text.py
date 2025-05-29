from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
import nltk
nltk.download('punkt')
import os
import sys
import faiss
import torch
import pandas as pd
import json
import time
import random
import re
from nltk.corpus import stopwords
# nltk.download('stopwords')


from mmseq_utils import align_and_analyze, ident_feature_correlated
from collections import OrderedDict
from feature_sim import peer_knn_predict, tsne_analysis_train_test, knn_predict

result_file = open("all_results.txt", "a+")
os.environ['OPENBLAS_NUM_THREADS'] = '1'
# 
def extract_words(text):
    words = re.findall(r'\w+', text)
    words = [w.lower() for w in words if w]
    return words

def evaluation(lines, labels, meta_labels, task_name):
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    labels = [l.strip() for l in labels]
    lines = [line.strip() for line in lines]
    
    assert len(labels) == len(lines) == len(meta_labels)
    
    total_exact_match = 0

    
    vis_meta_bleu_scores = []
    
    meteor_scores = []
    references = []
    hypotheses = []
    
    meta_references = []
    meta_hypotheses = []
    
    for pred, label, meta in tqdm(zip(lines, labels, meta_labels)):
        
        if pred.strip() == label.strip():
            total_exact_match += 1

        
        gt_tokens = tokenizer.tokenize(label, truncation=True, max_length=1024,
                                            padding='longest')
        ## added for galactica
        gt_tokens = list(filter(('<pad>').__ne__, gt_tokens))
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = tokenizer.tokenize(pred, truncation=True, max_length=1024,
                                            padding='longest')
        out_tokens = list(filter(('<pad>').__ne__, out_tokens))
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)
        
        meta_words = extract_words(meta)
        # meta_pred and meta_label
        pred_words = [word for word in extract_words(pred) if word in meta_words]
        meta_pred = " ".join(pred_words)
        
        label_words = [word for word in extract_words(label) if word in meta_words]
        meta_label = " ".join(label_words)

        gt_tokens = tokenizer.tokenize(meta_label, truncation=True, max_length=1024,
                                            padding='longest')

        gt_tokens = list(filter(('<pad>').__ne__, gt_tokens))
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = tokenizer.tokenize(meta_pred, truncation=True, max_length=1024,
                                            padding='longest')
        out_tokens = list(filter(('<pad>').__ne__, out_tokens))
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        meta_references.append([gt_tokens])
        meta_hypotheses.append(out_tokens)
        
        vis_meta_bleu_scores.append(corpus_bleu([[gt_tokens]], [out_tokens], weights=(.5,.5)))

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    
    bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))
    bleu2 *= 100
    bleu4 *= 100
    
    print('BLEU-2 score:', bleu2, file=result_file)
    print('BLEU-4 score:', bleu4, file=result_file)
    
    # meta-bleu calculation
    bleu2 = corpus_bleu(meta_references, meta_hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(meta_references, meta_hypotheses, weights=(.25,.25,.25,.25))
    bleu2 *= 100
    bleu4 *= 100
    print('Meta-BLEU-2 score:', bleu2, file=result_file)
    print('Meta-BLEU-4 score:', bleu4, file=result_file)
    
    _meteor_score = np.mean(meteor_scores)
    _meteor_score *= 100
    print('Average Meteor score:', _meteor_score, file=result_file)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for gt, out in tqdm(zip(labels, lines)):
        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    print('ROUGE score:', file=result_file)
    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]) * 100
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]) * 100
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]) * 100
    print('rouge1:', rouge_1, file=result_file)
    print('rouge2:', rouge_2, file=result_file)
    print('rougeL:', rouge_l, file=result_file)
    print("Exact Match:", total_exact_match / len(lines), file=result_file)
        
    with open(f"{task_name}_meta_list.txt", "w") as f:
        for i in range(len(vis_meta_bleu_scores)):
            f.write(f"{vis_meta_bleu_scores[i]}\n")

def UniProtQA():
    TRAIN_JSON = "/home/wujuntong/MinSimPro/ProteinText/UniProtQA/train.json"
    TEST_JSON = "/home/wujuntong/MinSimPro/ProteinText/UniProtQA/test.json"
    
    train_dic = json.load(open(TRAIN_JSON, "r"))
    test_dic = json.load(open(TEST_JSON, "r"))

    all_pred_labels, all_test_labels, all_percent = [], [], []
    for task in ['What is the function of this protein?',
                 'What are the official names of this protein?',
                 'What is the protein family that this protein belongs to?',
                 'What are the subcellular locations of this protein?']:
        
        train_seqs = []
        train_labels = []
        for k, v in tqdm(train_dic.items()):
            for qa in v["data"]:
                if qa[0] == task:
                    train_seqs.append(v["sequence"])
                    train_labels.append(qa[1])
        
        test_seqs = []
        test_labels = []
        for k, v in tqdm(test_dic.items()):
            for qa in v["data"]:
                if qa[0] == task:
                    test_seqs.append(v["sequence"])
                    test_labels.append(qa[1])
        
        print (f"\n MMSeqs retrieval annots for {task} ...", file=result_file)
        now_pred_labels, now_test_labels, now_percent = align_and_analyze(train_seqs, test_seqs, train_labels, test_labels, task)
        all_pred_labels.extend(now_pred_labels)
        all_test_labels.extend(now_test_labels)
        all_percent.append(now_percent)
        
    evaluation(all_pred_labels, all_test_labels, np.mean(all_percent))    

def Swiss_Prot():
    TRAIN_PATH= "/home/wujuntong/MinSimPro/ProteinText/OntoProteinDatasetV2/train.json"
    TEST_PATH = "/home/wujuntong/MinSimPro/ProteinText/OntoProteinDatasetV2/test.json"
    
    train_dic = json.load(open(TRAIN_PATH, "r"))
    test_dic = json.load(open(TEST_PATH, "r"))

    train_seqs = [item["sequence"] for item in train_dic]
    train_labels = [item["description"] for item in train_dic]

    test_seqs = [item["sequence"] for item in test_dic]
    test_labels = [item["description"] for item in test_dic]

    print("\n MMSeqs retrieval annots for OntoProteinDatasetV2 ...", file=result_file)
    pred_labels, test_labels = align_and_analyze(train_seqs, test_seqs, train_labels, test_labels, "OntoProteinDatasetV2")
    evaluation(pred_labels, test_labels, test_labels, "OntoProteinDatasetV2")
    


if __name__ == "__main__":

    # Swiss_Prot()
    # UniProtQA()
    # exit()
    
    JSON_FOLDER = "/home/wujuntong/MinSimPro/ProteinText/mol-inst-newsplit"
    
    all_train_seqs = []
    all_train_labels = []
    all_test_seqs = []
    all_test_labels = []
    all_meta_list = []
    
    all_train_features = []
    all_test_features = []
    
    task_ranges = []  # 新增：记录每个任务的test数据在all_test_seqs中的起止索引
    task_names = []
    for p in os.listdir(JSON_FOLDER):
        if not p.endswith("new_OOD_OOD.json") : continue
        # if p.endswith("new.json") : continue
        if "design" in p : continue

        print("task: ", p[:-5])
        print(f"\n now task: {p}", file=result_file)
        JSON_PATH = os.path.join(JSON_FOLDER, p)
        dic = json.load(open(JSON_PATH, "r"))
        
        try:
            train_dic = [d for d in dic if d["metadata"]["split"] == "train"]
            test_dic = [d for d in dic if d["metadata"]["split"] == "test"]
            
            train_seqs = [d["input"][4:-4] for d in train_dic]
            train_labels = [d["output"] for d in train_dic]
            
            test_seqs = [d["input"][4:-4] for d in test_dic]
            test_labels = [d["output"] for d in test_dic]
            
            meta_list = [d["metadata"]["annots"] for d in test_dic]
        except Exception as e:
            train_dic = [d for d in dic if d["split"] == "train"]
            test_dic = [d for d in dic if d["split"] == "test"]
            
            train_seqs = [d["sequence"] for d in train_dic]
            train_labels = [d["description"] for d in train_dic]
            # "description"
            test_seqs = [d["sequence"] for d in test_dic]
            test_labels = [d["description"] for d in test_dic]

            meta_list = [d["metadata"] for d in test_dic]
        
        # for mixed evaluation
        start_idx = len(all_test_seqs)
        all_train_seqs.extend(train_seqs)
        all_train_labels.extend(train_labels)
        all_test_seqs.extend(test_seqs)
        all_test_labels.extend(test_labels)
        all_meta_list.extend(meta_list)
        end_idx = len(all_test_seqs)
        task_ranges.append((start_idx, end_idx))
        task_names.append(p[:-5])
        
        print(p, len(train_seqs), len(train_labels), len(test_seqs), len(test_labels))
        
        # test_rouge_list = open(f"save_dir_{p[:-5]}_Llama-3.2-1B-Instruct_{p[:-5]}_rouge_l.txt").readlines()
        # test_rouge_list = [float(x.strip()) for x in test_rouge_list]
        
        # print("tsne analysis ...")
        # st_time = time.time()
        # # tsne_analysis_rouge_score(test_seqs=test_seqs, test_rouges=test_rouge_list, task_name=p[:-5])
        # random.shuffle(train_seqs)
        # random.shuffle(test_seqs)
        # tsne_analysis_train_test(train_seqs[:2000] , train_labels, test_seqs[:2000], test_labels, p[:-5])
        # print("tsne analysis time: ", time.time() - st_time)

        print(f"\n ESM feature retrieval for {p[:-5]} ...", file=result_file)
        pred_labels, test_labels, acc = peer_knn_predict(train_seqs, test_seqs, train_labels, test_labels, p[:-5])
        
        print(f"\n MMSeqs retrieval annots for {p[:-5]} ...", file=result_file)
        pred_labels, test_labels = align_and_analyze(train_seqs, test_seqs, train_labels, test_labels, p[:-5])
        
        # all_train_features.append(np.load(f"{p[:-5]}_train_features.npy"))
        # all_test_features.append(np.load(f"{p[:-5]}_test_features.npy"))
        
    # evaluation(pred_labels, test_labels, meta_list)
    
    # all_train_features = np.concatenate(all_train_features, axis=0)
    # all_test_features = np.concatenate(all_test_features, axis=0)
    
    # all_pred_labels = knn_predict(all_train_features, all_train_labels, all_test_features, k=1)
        
    print (f"\n MMSeqs retrieval annots for all_task ...", file=result_file)
    all_pred_labels, all_test_labels = align_and_analyze(all_train_seqs, all_test_seqs, all_train_labels, all_test_labels, 'all')
    
    # 按任务分开评测
    for (start, end), task_name in zip(task_ranges, task_names):
        task_name += "_all"
        print(f"\nEvaluation for task: {task_name}", file=result_file)
        evaluation(
            all_pred_labels[start:end],
            all_test_labels[start:end],
            meta_labels=all_meta_list[start:end],
            task_name=task_name
        )
    
    # print(f"\n ESM feature retrieval for all_task ...", file=result_file)
    # pred_labels, test_labels, acc = peer_knn_predict(all_train_seqs, all_test_seqs, all_train_labels, all_test_labels, "all")
    
    
