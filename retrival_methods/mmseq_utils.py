import os
import json
import Bio.Align
import Bio.pairwise2
import pandas as pd
import Bio
from rouge_score import rouge_scorer
import numpy as np
from scipy.stats import binned_statistic
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d


def group_by_seq(all_seqs, all_labels, all_instruction, all_meta, min_seq_id):
    
    with open("all_seqs.fasta", "w") as f:
        for i, seq in enumerate(all_seqs):
            f.write(f">seq_{i}\n{seq}\n")
    
    os.system(f"mmseqs easy-cluster -v 0 --cluster-mode 0 -c 0 -e 1e5 --single-step-clustering --min-seq-id {min_seq_id} all_seqs.fasta result tmp")
    
    cluster_file = f"result_cluster.tsv"

    tsv = pd.read_csv(cluster_file, sep="\t", header=None)

    cluster_id = 0
    las = ""
    clusters = []
    for line in tsv[[0, 1]].values:
        if line[0] != las: 
            cluster_id += 1
            las = line[0]
            clusters.append([])
        clusters[cluster_id - 1].append(line[1])
    
    print("Clusters 数目: ", len(clusters))
    
    
    # 按cluster中包含的序列数目排序
    clusters = sorted(clusters, key=lambda x: len(x))

    # 从小到大累加测试集，直到总数达到2000
    test_cluster = []
    test_count = 0
    for cluster in clusters:
        if test_count + len(cluster) <= 10000:
            test_cluster.append(cluster)
            test_count += len(cluster)
        else:
            break

    # 剩余的为训练集
    train_cluster = [c for c in clusters if c not in test_cluster]

    # 输出训练集的大小以及占数据集的比例
    test_size = sum(len(cluster) for cluster in test_cluster)
    total_size = sum(len(cluster) for cluster in clusters)
    print(f"测试集大小: {test_size}")
    print(f"测试集占比: {test_size / total_size:.4f}")
    
    # train_cluster = clusters[: int(len(clusters) * 0.8)]
    # test_cluster = clusters[int(len(clusters) * 0.8):]  

    train_seqs, test_seqs, train_instruction, train_meta = [], [], [], []
    train_labels, test_labels, test_instruction, test_meta = [], [], [], []
    
    for cluster in train_cluster:
        for seq in cluster:
            # 使用 "|" 分割序列信息
            parts = seq.split("_")
            seq_id = int(parts[1])  # 提取序列ID
            train_seqs.append(all_seqs[seq_id])
            train_labels.append(all_labels[seq_id])
            train_instruction.append(all_instruction[seq_id])
            train_meta.append(all_meta[seq_id])
    
    for cluster in test_cluster:
        for seq in cluster:
            # 使用 "|" 分割序列信息
            parts = seq.split("_")
            seq_id = int(parts[1])
            test_seqs.append(all_seqs[seq_id])
            test_labels.append(all_labels[seq_id])
            test_instruction.append(all_instruction[seq_id])
            test_meta.append(all_meta[seq_id])
    
    return train_seqs, test_seqs, train_labels, test_labels, train_instruction, test_instruction, train_meta, test_meta
    


def align_and_analyze(train_seqs, test_seqs, train_labels, test_labels, task_name):
    
    with open("train_seqs.fasta", "w") as f:
        for i, seq in enumerate(train_seqs):
            f.write(f">train_{i}\n{seq}\n")
    
    with open("test_seqs.fasta", "w") as f:
        for i, seq in enumerate(test_seqs):
            f.write(f">test_{i}\n{seq}\n")
    
    os.system("mmseqs easy-search --max-accept 1 -e 1e5 -v 0 test_seqs.fasta train_seqs.fasta result.m8 tmp")
    
    tsv = pd.read_csv("result.m8", sep="\t", header=None)

    # output_file = f"{task_name}_ali_res_with_info.txt"
    
    c0, c1 = 0, 0
    
    query_funs, target_funs = [], []
    order_query_fun = {}
    
    not_sim_ids = []
    
    # with open(output_file, "w") as f:
        
    #     for query, target in tsv[[0, 1]].values:
            
    #         # print(query, target)
            
    #         query_seq, query_fun = test_seqs[int(query.split("_")[1])], test_labels[int(query.split("_")[1])]
    #         target_seq, target_fun = train_seqs[int(target.split("_")[1])], train_labels[int(target.split("_")[1])]

    #         order_query_fun[int(query.split("_")[1])] = target_fun
            
    #         f.write(f"Query  Sequence: {query_seq} \n")
    #         f.write(f"Target Sequence: {target_seq} \n")
            
    #         align_result = Bio.Align.PairwiseAligner().align(query_seq, target_seq)
    #         f.write(f"Alignment Result: {align_result}\n")
            
            
    #         f.write(f"Query  Function: {query_fun} \n")
    #         f.write(f"Target Function: {target_fun}\n")
            
    #         f.write(f"Simularity: {query_fun} vs {target_fun}\n")
    #         if query_fun == target_fun: 
    #             c1 += 1
    #         else:
    #             not_sim_ids.append(int(query.split("_")[1]))
    #         c0 += 1

    #         f.write("---------------\n")

    # print(f"Average Similarity: {sum(avg_rg) / len(avg_rg):.4f}")
    print(f"Total Similarity: {c1/c0:.4f}")
    
    missed_num = 0
    with open(f"{task_name}_predicted.tsv", "w") as f:
        f.write("seq\toutput\n")
        for i in range(len(test_labels)):
            target_funs.append(test_labels[i])
            try:
                f.write(f"{test_seqs[i]}\t{order_query_fun[i]}\n")
                query_funs.append(order_query_fun[i])
            except Exception as e:
                f.write(f"{test_seqs[i]}\tNot Matched\n")
                query_funs.append("Not Matched")
                missed_num += 1
                # print(f"test_{i} Not Matched",end='; ')
                
    # percent = (len(test_labels) - missed_num) / len(test_labels)
    
    print("Retrieval Percent: ", (len(test_labels) - missed_num) / len(test_labels))
    
    # return not_sim_ids
    
    return query_funs, target_funs
    

def ident_feature_correlated(train_seqs, test_seqs, train_labels, test_labels, task_name):
    
    with open("train_seqs.fasta", "w") as f:
        for i, seq in enumerate(train_seqs):
            f.write(f">train_{i}\n{seq}\n")
    
    with open("test_seqs.fasta", "w") as f:
        for i, seq in enumerate(test_seqs):
            f.write(f">test_{i}\n{seq}\n")
    
    os.system("mmseqs easy-search -e 1e5 -v 0 --threads 128 test_seqs.fasta train_seqs.fasta result.m8 tmp")
    
    tsv = pd.read_csv("result.m8", sep="\t", header=None)

    identity_bins = np.linspace(0, 1, 21)
    bin_indices = np.digitize(tsv[2], identity_bins, right=True)
    bin_correct = np.zeros(len(identity_bins))
    bin_total = np.zeros(len(identity_bins))

    for query, target, identity in tqdm(tsv[[0, 1, 2]].values):
        idx = np.digitize(identity, identity_bins, right=True) - 1
        query_fun = test_labels[int(query.split("_")[1])]
        target_fun = train_labels[int(target.split("_")[1])]
        bin_total[idx] += 1
        if query_fun == target_fun:
            bin_correct[idx] += 1
    
    prob = np.divide(bin_correct, bin_total, out=np.zeros_like(bin_correct), where=bin_total!=0)
    print(prob)
    
    bin_centers = (identity_bins[:-1] + identity_bins[1:]) / 2

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, prob[:-1], width=identity_bins[1]-identity_bins[0], alpha=0.6, color='skyblue', edgecolor='black', label='Probability')
    smooth_prob = gaussian_filter1d(prob[:-1], sigma=2)
    plt.plot(bin_centers, smooth_prob, color='red', linewidth=2, label='Smoothed trend')
    plt.xlabel("Sequence Identity")
    plt.ylabel("P(query_fun == target_fun)")
    plt.title("Probability of Function Match vs. Sequence Identity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{task_name}_identity_vs_fun_prob.png")
    # points = []

    print()
    
    # for query, target, identity in tqdm(tsv[[0, 1, 2]].values):
        
    #     query_fun = test_labels[int(query.split("_")[1])]
    #     target_fun = train_labels[int(target.split("_")[1])]
        
    #     if query_fun == target_fun:
    #         points.append(identity)
    #     # scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    #     # score = scorer.score(str(query_fun), str(target_fun))
    #     # rouge_l = score['rougeL'].fmeasure
    #     # rouge_l = (query_fun == target_fun)
        
    #     # points.append((identity, rouge_l))

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 6))
    # plt.hist(points, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    # plt.xlabel("Identity")
    # plt.ylabel("Count")
    # plt.title("Distribution of Identity Values in Points")
    # plt.tight_layout()
    # plt.savefig(f"{task_name}_identity_distribution.png")
    # 拆分 identity 和 rouge_l
    # x = [float(p[0]) for p in points]
    # y = [float(p[1]) for p in points]

    # plt.figure(figsize=(8, 6))
    # plt.scatter(x, y, alpha=0.3, label="Data points")

    # # 计算平滑曲线（分箱平均）
    # bins = np.linspace(min(x), max(x), 20)
    # bin_means, bin_edges, _ = binned_statistic(x, y, statistic='mean', bins=bins)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # plt.plot(bin_centers, bin_means, color='red', linewidth=2, label="Mean trend")

    # plt.xlabel("Sequence Identity")
    # plt.ylabel("ROUGE-L F1")
    # plt.title("Identity vs. ROUGE-L F1")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"{task_name}_identity_vs_rouge.png")



if __name__ == "__main__":

    for task in ["protein_function.json",
                "general_function.json",
                "domain_motif.json",
                "catalytic_activity.json"]:
        for msi in [0.0]:
            
            JSON_PATH = "/home/wujuntong/MinSimPro/ProteinText/Protein-oriented_Instructions/" + task
            dic = json.load(open(JSON_PATH, "r"))
            seqs = [d["input"][4:-4] for d in dic]
            labels = [d["output"] for d in dic]
            insts = [d["instruction"] for d in dic]
            meta = [d["metadata"]["annots"] for d in dic]
            
            print(f"task: {task}")
            print(f"All seqs: {len(seqs)}")
            print(f"Now id: {msi}")
            train_seqs, test_seqs, train_labels, test_labels, train_instruction, test_instruction, train_meta, test_meta   = \
                group_by_seq(seqs, labels, insts, meta, min_seq_id=msi)
            
            train_dic = [{"instruction": ins, "sequence": seq, "description": lab, "split": "train", "metadata": meta} for (ins,seq,lab,meta) in zip(train_instruction, train_seqs, train_labels, train_meta) ]
            test_dic = [{"instruction": ins, "sequence": seq, "description": lab, "split": "test", "metadata": meta} for (ins,seq,lab,meta) in zip(test_instruction, test_seqs, test_labels, test_meta) ]
            
            dic = train_dic + test_dic
            with open(f"{task[:-5]}_msi_{msi}_new.json", 'w') as f:
                json.dump(dic, f, indent=4)
            
            # ident_feature_correlated(train_seqs, test_seqs, train_labels, test_labels, task[:-5] + f"_msi_{msi}")