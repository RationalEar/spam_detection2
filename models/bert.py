import torch
import torch.nn as nn
from transformers import BertModel


class SpamBERT(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=1, dropout=0.1):
        super(SpamBERT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        probs = torch.sigmoid(logits).squeeze(-1)
        return probs, outputs.attentions if hasattr(outputs, 'attentions') else None

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
