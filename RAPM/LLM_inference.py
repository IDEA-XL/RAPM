## by lihao
## DeepSeek-v3 inference code for Pubchem-QA dataset
import json
from tqdm import tqdm
import random

from openai import OpenAI
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import numpy as np
from transformers import AutoTokenizer
import re, os
import sys

result_file = open("all_results.txt", "a+")
os.environ['OPENBLAS_NUM_THREADS'] = '1'

def extract_words(text):
    words = re.findall(r'\w+', text)
    words = [w.lower() for w in words if w]
    return words

def evaluation(lines, labels, meta_labels):
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    labels = [l.strip() for l in labels]
    lines = [line.strip() for line in lines]
    
    assert len(labels) == len(lines) == len(meta_labels)
    
    total_exact_match = 0
    
    
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

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)
    
    bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))
    bleu2 *= 100
    bleu4 *= 100
    
    print('BLEU-2 score:', bleu2, file=result_file)
    print('BLEU-4 score:', bleu4, file=result_file)
    
    bleu2 = corpus_bleu(meta_references, meta_hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(meta_references, meta_hypotheses, weights=(.25,.25,.25,.25))
    bleu2 *= 100
    bleu4 *= 100
    print('Meta-BLEU-2 score:', bleu2, file=result_file)
    print('Meta-BLEU-4 score:', bleu4, file=result_file)
    print('Meta-BLEU-2 score:', bleu2)
    print('Meta-BLEU-4 score:', bleu4)
    
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
    print('rougeL:', rouge_l)
    print("Exact Match:", total_exact_match / len(lines), file=result_file)



client = OpenAI(
    api_key="your_api_key",
    base_url="your_base_url",
)


def api_inference(RAG_prompt, model):

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", 
             "content": RAG_prompt,
            },
        ],
        stream=False,
        timeout=48000,
        temperature=0.0,
    )
    output_results = response.choices[0].message.content

    output_results = output_results.replace("```json", "")
    output_results = output_results.replace("```", "")

    try:
        result_json = json.loads(output_results)
        if "description" in result_json:
            output_results = result_json["description"]
        else:
            print("Error: 'description' key not found in JSON. Returning original output.")
    except json.JSONDecodeError:
        print("Error: Output is not valid JSON. Returning original output.")
        print("Output:", output_results)
    return output_results


if __name__ == "__main__":

    now_task = sys.argv[1]
    now_k = int(sys.argv[2])
    model = "gpt-4.1"
    
    result_file = open("api_evaluation_results.txt", "w")
    all_input_prompt_len = 0
    infer_numbers = 256
    
    print("now task: ", now_task, "now k: ", now_k)
    print("now task: ", now_task, "now k: ", now_k, file=result_file)
    JSON_PATH = f"{now_task}_RAP_Top_{now_k}.json"
    dic = json.load(open(JSON_PATH, "r"))
    all_answers = []
    all_labels = []
    all_meta_labels = []

    for i in tqdm(range(infer_numbers)):
        d = dic[i]
        RAG_prompt = d['RAG_prompt']
        answer = d['labels']
        meta_answer = d["meta_label"]

        LLM_answer = api_inference(RAG_prompt, model=model)

        all_input_prompt_len += len(RAG_prompt.split())
        
        all_answers.append(LLM_answer)
        all_labels.append(answer)
        all_meta_labels.append(meta_answer)

    print("Average input prompt length: ", all_input_prompt_len / infer_numbers)
    
    evaluation(all_answers, all_labels, all_meta_labels)

    for i in range(len(all_answers)):
        info_dict = {
            "RAG_prompt": dic[i]['RAG_prompt'],
            "answer": all_answers[i],
            "label": all_labels[i],
            "meta_label": all_meta_labels[i]
        }

        with open(f"RAPM_{now_task}_{now_k}_results.json", 'a+') as f:
            json.dump(info_dict, f)
            f.write('\n')

        