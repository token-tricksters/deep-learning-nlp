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

class BertForMaskedLM(nn.Module):
    def __init__(self, vocab_size):
        super(BertForMaskedLM, self).__init__()
        self.base_model = BertModel.from_pretrained('bert-base-uncased')
        self.predictor = nn.Linear(768, vocab_size)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask)
        predictions = self.predictor(outputs["last_hidden_state"])
        return predictions

# Initialize Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
model = BertForMaskedLM(tokenizer.vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Hyperparameters
MASK_PROB = 0.15
VOCAB_SIZE = tokenizer.vocab_size

# MLM Loss
criterion = nn.CrossEntropyLoss()

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

    # 3. Get BERT's output
    predictions = model(input_ids, attention_mask)

    masked_predictions = predictions[masked_indices]
    masked_token_ids = original_ids[masked_indices]

    mlm_loss = criterion(masked_predictions, masked_token_ids)
    
    # Zero out previous gradients
    optimizer.zero_grad()

    # 4. Calculate the loss and perform backpropagation
    mlm_loss.backward()

    # 5. Optimization step
    optimizer.step()

    print(f"MLM Loss: {mlm_loss.item()}")

    # mini-eval

    
    inp = X[:1]

    encoding = tokenizer(inp, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    probability_matrix = torch.full(input_ids.shape, MASK_PROB)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    original_ids = input_ids.clone()
    input_ids[masked_indices] = tokenizer.convert_tokens_to_ids('[MASK]')

    model.eval()
    with torch.no_grad():
        out = model(input_ids, attention_mask)

    print(tokenizer.convert_ids_to_tokens(input_ids[0]))
    
    out = torch.argmax(out, axis=2)
    print(tokenizer.convert_ids_to_tokens(out[0]))