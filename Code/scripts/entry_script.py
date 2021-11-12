"""
Created in November 2021

Entry script which handles and processes the request during inference

@author: Martin Danner
@company: scieneers GmbH
@mail: martin.danner@scieneers.de
"""


from transformers import AutoTokenizer
import numpy as np
import torch
import json
import os


def init():
    global model, device
    model_path = os.getenv('AZUREML_MODEL_DIR')
    model_file = os.listdir(model_path)[0]
    model_file_path = os.path.join(model_path, model_file)
    model = torch.load(model_file_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)


def run(request):
    data = json.loads(request)
    token_list = tokenize(data['review'], device)
    output = model.forward(token_list[0], token_list[1], token_list[2])
    logits_detached = output.logits.cpu().detach().numpy() 
    logits_soft_max = soft_max(logits_detached)
    response = {'positive_score': str(logits_soft_max[0][0]), 'negative_score': str(logits_soft_max[0][1])}

    return json.dumps(response)
    
def tokenize(film_review: str, device):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokens = tokenizer(film_review, padding="max_length", truncation=True, return_tensors='pt')
    inputs = tokens.input_ids.to(device=device)
    att_mask = tokens.attention_mask.to(device=device)
    ids = tokens.token_type_ids.to(device=device)  
    token_list = [inputs, att_mask, ids]

    return token_list

def soft_max(a):
    return np.exp(a)/np.sum(np.exp(a))

    