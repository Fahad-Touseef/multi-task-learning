import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# Hypothetical Dataset
class MTLDataset(Dataset):
    def __init__(self, data):
        """
        data: list of dicts with keys:
          - 'sentence': str
          - 'sent_label': int
          - 'token_labels': List[int]   
        """
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        item = self.data[i]
        return {
            'sentence'     : item['sentence'],
            'sent_label'   : item['sent_label'],
            'token_labels' : torch.tensor(item['token_labels'], dtype=torch.long)
        }

# Collate function for DataLoader
def collate_fn(batch, tokenizer):
    # Tokenize sentences and pad them
    sentences = [b['sentence'] for b in batch]
    tokenized = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    sent_labels = torch.tensor([b['sent_label'] for b in batch], dtype=torch.long)

    # Align token_labels with tokenized input
    token_labels = []
    for i, b in enumerate(batch):
        word_ids = tokenized.word_ids(batch_index=i)  # Map tokens to original words
        labels = b['token_labels']
        aligned_labels = []
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)  # Padding token
            elif word_id < len(labels):
                aligned_labels.append(labels[word_id])
            else:
                aligned_labels.append(-100)  # Special tokens like [CLS], [SEP]
        token_labels.append(torch.tensor(aligned_labels, dtype=torch.long))

    # Stack token_labels into a single tensor
    token_labels = torch.stack(token_labels, dim=0)

    return tokenized, sent_labels, token_labels
