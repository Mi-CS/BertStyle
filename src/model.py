import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

class MLPLayer(nn.Module):
    """
    MLP Head with a hidden layer to take BERT [CLS] representation 
    to the contrastive learning space. 
    """
    def __init__(self):
        super().__init__()
        self.dense_hidden = nn.Linear(768, 1536)
        self.dense_out = nn.Linear(1536, 768)

    def forward(self, features):
        x = self.dense_hidden(features)
        x = self.dense_out(F.relu(x))
        return x
    
class BertMLP(nn.Module): 
    """
    BERT model with a MLP Head. Assumes the BERT model given when 
    initialization has the same arguments as BERT from HuggingFace. 
    """
    
    def __init__(self, bert_model, mlp):
        super().__init__()
        self.bert = bert_model
        self.mlp = mlp
    
    def forward(self, x_input, x_att):
        x = self.bert(input_ids = x_input, attention_mask = x_att)
        x = x.last_hidden_state[:, 0]
        x = self.mlp(x)
        return x        
    
class Similarity(nn.Module):
    """
    Cosine similarity with a temperature parameter.
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp



def init_model(device):
    """
    Initialize the model using a pretrained Bert base uncased 
    from the transformers (HuggingFace) library. 
    """
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    model = BertMLP(bert_model = bert_model, 
                    mlp = MLPLayer())

    # Ensure that BERT weights are trainable 
    for param in model.bert.parameters(): 
        param.requires_grad = True 
    
    if device.type == "cuda": 
        model = nn.DataParallel(model) # Use all GPUs
        model.to(device)
    
    return model