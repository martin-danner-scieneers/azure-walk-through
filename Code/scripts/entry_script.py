
from azureml.core.model import Model
from azure.storage.blob import BlockBlobService
from transformers import AutoTokenizer
import numpy as np
import torch
import json
import pickle

def init():
    
    global model, device
    
    block_blob_service = BlockBlobService(connection_string='DefaultEndpointsProtocol=https;AccountName=dlrundstedt;AccountKey=xQ1JHBRC4h0A8mWDIWP5AbKGSxJHa8qbLA2XR0QjrQjEDBf/IOLA9zfrQr6ojoZioK1Z07EE4W8OB93ttujhuw==;EndpointSuffix=core.windows.net')

    blob_item = block_blob_service.get_blob_to_bytes('models','runtime_params.pkl')

    params = pickle.load(blob_item.content)

    model_path = Model.get_model_path(params['model_name'])

    model = torch.load(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)


def run(request):
    data = json.loads(request)
    token_list = tokenize(data['review'])
    output = model.forward(token_list[0], token_list[1], token_list[2])
    logits_detached = output.logits.cpu().detach().numpy() 
    logits_soft_max = soft_max(logits_detached)
    response = {'positive_score': logits_soft_max[0], 'negative_score': logits_soft_max[1]}

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