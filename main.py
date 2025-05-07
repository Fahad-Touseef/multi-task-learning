import torch
from model import SentenceTransformer, MultiTaskTransformer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from data import MTLDataset, collate_fn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW



def test_sentence_transformer():
    model = SentenceTransformer()

    sentences = [
        "I enjoy playing table tennis.",
        "I have applied for a job at Fetch.",
        "The job is a ML Apprentice role.",
        "I am excited about the opportunity.",
    ]

    with torch.no_grad():
        model_input = model.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        embeddings = model(model_input)
    print("Embeddings shape:", embeddings.shape)
    
    embeddings_np = embeddings.numpy()

    # Reduce to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=2)
    reduced = tsne.fit_transform(embeddings_np)
    # Alternatively, we can use PCA for dimensionality reduction
    # pca = PCA(n_components=2)
    # reduced = pca.fit_transform(embeddings_np)


    # Plot
    plt.figure(figsize=(8, 6))
    for i, sentence in enumerate(sentences):
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.text(x+1, y+1, sentence, fontsize=9)

    plt.title("Sentence Embeddings Visualized with t-SNE")
    plt.grid(True)
    plt.show()    

def train_multi_task_model(num_epochs=2, batch_size=2, lr=5e-5):
    # Sample data
    # Sentence labels: 
    # 0 - Sports, 1 - Jobs, 2 - Travel
    # NER labels:
    # 0 - Person, 1 - Location, 2 - Organization, 3 - Other
    data = [
        {
            'sentence': "I enjoy playing table tennis.",
            'sent_label': 0,  # Sports
            'token_labels': [3, 3, 3, 3, 3]  # All tokens are 'Other'
        },
        {
            'sentence': "I have applied for a job at Fetch.",
            'sent_label': 1,  # Jobs
            'token_labels': [3, 3, 3, 3, 3, 2]  # 'Fetch' is an Organization
        },
        {
            'sentence': "I visited Paris last summer.",
            'sent_label': 2,  # Travel
            'token_labels': [3, 3, 1, 3, 3]  # 'Paris' is a Location
        }
        # Add more data
    ]

    dataset = MTLDataset(data)
    model = MultiTaskTransformer()
    model.train()

    # Freeze the BERT backbone
    for param in model.bert.parameters():
        param.requires_grad = False

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    ner_ignore_index = -100  # Ignore index for padding in token labels (same as in collate_fn)

    # Pass the tokenizer to the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, model.tokenizer),
        shuffle=True,
        drop_last=True
    )

    for epoch in range(num_epochs):
        total_loss = 0.0

        # Metrics accumulators
        sent_correct = 0
        sent_total = 0
        ner_correct = 0
        ner_total = 0

        for batch in dataloader:
            model_input, sent_labels, token_labels = batch  # Unpack the tuple

            outputs = model(model_input)

            # Compute losses
            loss_sent = F.cross_entropy(outputs['sentence_logits'], sent_labels)

            ner_logits_flat = outputs['ner_logits'].view(-1, outputs['ner_logits'].size(-1))  # Flatten for loss computation
            token_labels_flat = token_labels.view(-1)  # Flatten to match ner_logits_flat
            loss_ner = F.cross_entropy(
                ner_logits_flat,
                token_labels_flat,
                ignore_index=ner_ignore_index  # Ignore padding tokens
            )

            loss = loss_sent + loss_ner  # May use a weighted sum instead

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update metrics
            pred_sent = outputs['sentence_logits'].argmax(dim=1)
            sent_correct += (pred_sent == sent_labels).sum().item()
            sent_total += sent_labels.size(0)

            pred_ner = ner_logits_flat.argmax(dim=1)
            mask = token_labels_flat != ner_ignore_index
            ner_correct += (pred_ner[mask] == token_labels_flat[mask]).sum().item()
            ner_total += mask.sum().item()

        avg_loss = total_loss / len(dataloader)  # Divide by the number of batches in the dataloader
        sent_acc = sent_correct / sent_total
        ner_acc = ner_correct / ner_total

        print(f"Epoch {epoch+1:02d} ─ loss: {avg_loss:.4f}"
              + f" │ SentAcc: {sent_acc:.3f}"
              + f" │ NERAcc: {ner_acc:.3f}")


if __name__ == "__main__":
    # test_sentence_transformer()
    train_multi_task_model()