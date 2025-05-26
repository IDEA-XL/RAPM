

<h2 align="center">
  <img src="figs/protein.png" style="vertical-align:middle; width:23px; height:23px;" />
  <a href=""> RAPM (Retrieval-Augmented Protein Modeling) </a>
</h2>

**Official implementation for the paper "Rethinking Text-based Protein Understanding: Retrieval or LLM?"**

### [📖] Abstract:

In recent years, protein-text models have gained significant attention for their potential in protein generation and understanding. Current approaches focus on integrating protein-related knowledge into large language models through continued pretraining and multi-modal alignment, enabling simultaneous comprehension of textual descriptions and protein sequences. 
Through a thorough analysis of existing model architectures and text-based protein understanding benchmarks, **we identify significant data leakage issues present in current benchmarks.** Moreover, **conventional metrics derived from natural language processing fail to accurately assess the model's performance in this domain.** To address these limitations, we reorganize existing datasets and introduce a novel evaluation framework based on biological entities. Motivated by our observation, we propose a **retrieval-enhanced method**, which significantly outperforms fine-tuned LLMs for protein-to-text generation and shows accuracy and efficiency in training-free scenarios.

![alt text](figs/main_fig.png)

### [‼️] Data Leakage in Existing Protein-to-Text Benchmark 

We evaluated four commonly used benchmarks in the field of text-based protein understanding, including the protein comprehension tasks (Function, Description, Domain, and Catalytic) from Mol-Instructions [1], UniProtQA [2], the Swiss-Prot Protein Caption dataset [3], and the ProteinKG25 dataset [4].  

<p style="font-style: italic; font-size: smaller;">
[1] Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models <br>
[2] BioMedGPT: Open Multimodal Generative Pre-trained Transformer for BioMedicine <br>
[3] ProtT3: Protein-to-Text Generation for Text-based Protein Understanding <br>
[4] OntoProtein: Protein Pretraining With Ontology Embedding <br>
</p>

For sequence retrieval, we employed MMSeqs2 with the following command:  
```sh
mmseqs easy-search --max-accept 1 -e 1e5 -v 0 test_seqs.fasta train_seqs.fasta result.m8 tmp  
```
For each protein sequence in the test set, the label of its most similar counterpart in the training set was assigned as the predicted output.  

The experimental results are presented in the table below, demonstrating that all current LLM-based models underperform compared to retrieval-based models. Furthermore, we conducted an analysis of data leakage rates. Here, we define leakage rate as the probability of obtaining identical labels when using the retrieval method (for Mol-Instructions, we only consider whether the metadata matches, without accounting for variations in response phrasing).

![alt text](figs/tab1.png)

