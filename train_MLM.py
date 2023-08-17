import torch
import torch.nn as nn
from bert import BertModel
from tokenizer import BertTokenizer
from torch.utils.data import Dataset, DataLoader


class QuoraTextDataset(Dataset):

    def __init__(self):
        # Load example data
        f = open("data/quora-train.csv", "r")
        sentences  = f.readlines() 
        self.sentences = [x.split("\t")[2] for x in sentences]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

quora_text_dataset = QuoraTextDataset()
dataloader = DataLoader(quora_text_dataset, batch_size=64, shuffle=True, num_workers=0)

# Initialize Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
model = BertModel.from_pretrained('bert-base-uncased')
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Hyperparameters
BATCH_SIZE = 1
MASK_PROB = 0.15
VOCAB_SIZE = tokenizer.vocab_size

# MLM Loss
criterion = nn.MSELoss()

for iter, X in enumerate(dataloader):

    # 1. Tokenize the sentence
    encoding = tokenizer(X, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # 2. Mask 15% of the tokens for MLM task
    # This approach randomly masks tokens. You can further enhance this.
    probability_matrix = torch.full(input_ids.shape, MASK_PROB)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    original_ids = input_ids.clone()
    input_ids[masked_indices] = tokenizer.convert_tokens_to_ids('[MASK]')

    embedding_unmasked = model.embed(input_ids)

    # 3. Get BERT's output
    outputs = model(input_ids, attention_mask)
    predictions = outputs["last_hidden_state"]

    # Enforce consistency on the unmasked input embeddings vs output embeddings
    mlm_loss = criterion(predictions, embedding_unmasked)
    
    # Zero out previous gradients
    optimizer.zero_grad()

    # 4. Calculate the loss and perform backpropagation
    mlm_loss.backward()

    # 5. Optimization step
    optimizer.step()

    print(f"MLM Loss: {mlm_loss.item()}")

