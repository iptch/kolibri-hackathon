import torch
from transformers import BertModel
from torch import nn

# Define the model
class SearchTermDescriptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output  # We use the [CLS] token embedding