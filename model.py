import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast


class SentenceTransformer(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(SentenceTransformer, self).__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, model_input):
        model_output = self.bert(**model_input)
        # Using CLS token (the first token)
        embeddings = model_output.last_hidden_state[:, 0]  
        
        # Alternatively, we can use mean pooling
        # embeddings = torch.mean(model_input['attention_mask'].unsqueeze(-1) * model_output.last_hidden_state, dim=1) 
        
        return embeddings # (batch_size, hidden_size)
    

class MultiTaskTransformer(nn.Module):
    def __init__(self,
                 backbone_name='bert-base-uncased',
                 num_sentence_classes=3,
                 num_ner_labels=4):
        super().__init__()
        # Shared backbone
        self.tokenizer = BertTokenizerFast.from_pretrained(backbone_name)
        self.bert = BertModel.from_pretrained(backbone_name)
        
        hidden_dim = self.bert.config.hidden_size
        
        # Task A: Sentence Classification head
        self.sent_classifier = nn.Linear(hidden_dim, num_sentence_classes)
        
        # Task B: NER head (token classification)
        self.ner_classifier  = nn.Linear(hidden_dim, num_ner_labels)
    
    def forward(self, model_input):
        model_outputs = self.bert(**model_input)
        
        # 1) Sentence-level embedding via CLS
        cls_emb = model_outputs.last_hidden_state[:, 0]    # (batch, hidden_dim)
        # 2) Token-level embeddings
        token_embs = model_outputs.last_hidden_state         # (batch, seq_len, hidden_dim)
        
        # 3) Task predictions
        sent_logits = self.sent_classifier(cls_emb)         # (batch, num_sentence_classes)
        ner_logits  = self.ner_classifier(token_embs)       # (batch, seq_len, num_ner_labels)
        
        return {
            'sentence_logits': sent_logits,
            'ner_logits'     : ner_logits
        }
