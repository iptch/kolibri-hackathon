from transformers import BertModel
import torch.nn as nn

class BertEmbedding(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased"):
        super(BertEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)

    def forward(self, input_ids, attention_mask):
        # We only need the outputs from the BERT model, not the pooled outputs
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]  
        return bert_output
