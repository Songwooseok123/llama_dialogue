import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
def prompt_make(path):
    with open(path, 'rb') as fr:
        data = pickle.load(fr)
    del_index = []
    for i in range(len(data)):
        if len(data['context'][i]) %2 ==0:
            del_index.append(i)
    data.drop(del_index, axis=0, inplace=True)
    data=data.reset_index(drop = True)
    data = data[:2]
    def make_prompt(j):
        sys_prompt = '[INST] <<SYS>>\nAssume you are user and continue the conversation.\n<</SYS>>\n'
        utter1 = data['context'][j][0] + '[/INST]\n'
        prompt = sys_prompt + utter1
        #prompt = utter1
        for i in range(len(data['context'][j])-1):
            
            if (i+1)%2 ==1 :
                #print("??")
                utter2 = data['context'][j][i+1] + '</s>'
            else: 
                #print("!@!")
                utter2 = '\n<s>[INST]'+data['context'][j][i+1] + '[/INST]'
            prompt = prompt + utter2
        return prompt
    prompt_gen = []
    for j in tqdm(range(len(data))):
        prompt_gen.append(make_prompt(j))
    data['prompt']=prompt_gen
    return data