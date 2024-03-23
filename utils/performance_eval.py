import nltk
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.nist_score import sentence_nist
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate
import numpy as np
from rouge import Rouge
from collections import Counter
import math,torch
from evaluate import load
import pandas as pd
from transformers import BitsAndBytesConfig,LlamaTokenizer,LlamaForCausalLM,GPT2Tokenizer, GPT2LMHeadModel
from bert_evaluation import Model,get_accuracy_by_bert
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from evaluate import logging
from transformers import BertTokenizerFast, BertModel
from torch import nn
import os,pickle

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        #참고로 'bert-base-uncased' 모델은 버트의 가장 기본적인 모델을 의미
        #uncased는 모든 문장을 소문자로 대체하겠다는 것 
        
        self.classifier  = nn.Sequential(
            nn.ReLU(),
            nn.Linear(768,4)
        )
        
        
    def forward(self,inputs):
        
        #print(inputs['input_ids'].shape)  
        bert_outputs = self.bert(**inputs,return_dict =True)
        #print("bert_outputs",bert_outputs)
        pooler_output = bert_outputs.last_hidden_state[:,0]
        # bert_outputs.last_hidden_state[:,0] -> 첫번째니까 [cls] 토큰의
        # embedding만을 뽑아내어 classification task를 위한 텐서로 변환한다? 
        #print("pooler_output",pooler_output.shape)
        
        logits = self.classifier(pooler_output)
        #print("logits",logits.shape)
        
        return logits

def cal_meteor(hypothesis,reference):
    meteor = evaluate.load('meteor')
    results = meteor.compute(predictions=hypothesis, references=reference)['meteor']
    return results
def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):
    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]
def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):

    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence
def distinct_n_sentence_level(sentence, n):
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)
def distinct_n_corpus_level(sentences, n):
   
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)

def calculate_rouge_l(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores[0]['rouge-l']['f']

def calculate_entropy_ngram(tokens, n):
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    ngram_counts = Counter(ngrams)
    total_ngrams = sum(ngram_counts.values())
    
    # Calculate entropy
    entropy = 0.0
    for count in ngram_counts.values():
        probability = count / total_ngrams
        entropy -= probability * math.log2(probability)
    
    return entropy

def calculate_entropy_2(tokens):
    return calculate_entropy_ngram(tokens, 2)

def calculate_entropy_4(tokens):
    return calculate_entropy_ngram(tokens, 4)

def perplexity_cal(hypothesis,tokenizer):
    data = [x for x in hypothesis if len(x)>0]
    batch_size= 16
    add_start_token= True, 
    device='cuda'
    max_length=None
    
    
    model_path ="meta-llama/Llama-2-7b-chat-hf"
    nf4_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_use_double_quant=True,
       bnb_4bit_compute_dtype=torch.bfloat16
    )
    model =LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path = model_path,
                                        #load_in_8bit=True, #  7.7GB로
                                            quantization_config =nf4_config, #  4.4GB로 
                                            device_map="auto", # gpu 꽉차면 cpu로 올려줌 
                                            
                                           )

    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})
    
    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length
    
    encodings = tokenizer(
        data,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)
    
    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]
    
    # check that each input is long enough:
    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."
    
    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")
    
    for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]
    
        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
            )
    
        labels = encoded_batch
    
        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits
    
        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()
    
        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )
    
        ppls += perplexity_batch.tolist()
    
    return round(np.mean(ppls),2)

def result(reference,hypothesis):
    model_path ="meta-llama/Llama-2-7b-chat-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    reference_sentences = reference
    hypothesis_sentences = hypothesis

    meteor =[]
    nist_2_scores = []
    nist_4_scores = []
    rouge_l_scores = []
    entropy_2_scores = []
    entropy_4_scores = []
    ppl  = perplexity_cal(hypothesis,tokenizer)
    meteorss = cal_meteor(hypothesis,reference)
    for ref, hyp in zip(reference_sentences, hypothesis_sentences):
        reference1 = tokenizer.encode(ref)
        hypothesis1 = tokenizer.encode(hyp)
        #print(ref,hypothesis1)
        
        if 1<len(hypothesis1):
            nist_2_scores.append(sentence_nist([reference1], hypothesis1,2))
        if 3<len(hypothesis1):
            nist_4_scores.append(sentence_nist([reference1], hypothesis1,4))    
        if hyp !='':
            rouge_l_scores.append(calculate_rouge_l(ref, hyp))

        entropy_2_scores.append(calculate_entropy_2(hyp))
        entropy_4_scores.append(calculate_entropy_4(hyp))
    bleu = load("bleu")
    results = bleu.compute(predictions=hypothesis_sentences, references=reference_sentences)
    bleu1 = results['precisions'][0]
    bleu2 = results['precisions'][1]
    return  [round(meteorss*100,2),
             round(distinct_n_corpus_level(hypothesis, 1)*100,2) ,
             round(distinct_n_corpus_level(hypothesis, 2)*100,2)  ,
             #round(distinct_n_corpus_level(hypothesis, 3)*100,2)  ,
             round(np.mean(nist_2_scores),2),
             round(np.mean(nist_4_scores),2),
             round(bleu1*100,2) ,
             round(bleu2*100,2) ,
             round(np.mean(rouge_l_scores)*100,2) ,
             round(np.mean(entropy_2_scores),2) ,
             round(np.mean(entropy_4_scores),2),
             round(ppl,2) ]


def result_per_setting(setting_name,device):
    
    with open('results/'+setting_name+'.pickle', 'rb') as fr:
        data = pickle.load(fr)
    #print(data.head(1))
    print("######Generated example 2개#######")
    for i in range(2):
        print(data['generated'][i])
    
    reference, generated = list(data['reference']),list(data['generated'])
    
    accuracy = get_accuracy_by_bert(data,device)
    
    rrresult= pd.DataFrame(columns=['acc','METEOR','D-1','D-2','N-2', 'N-4','BLEU1','BLEU2','RougeL','Entropy2', 'Entropy4','Perplexity'],index =[setting_name])
    rrresult.iloc[0] = [accuracy]+result(reference, generated)
    return rrresult