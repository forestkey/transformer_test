import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field
from torchtext.datasets import Multi30k


# Define the source and target language fields
SRC = Field(tokenize='spacy',
            tokenizer_language='de',
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

TRG = Field(tokenize='spacy',
            tokenizer_language='en',
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

# Load the Multi30k dataset
train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                    fields=(SRC, TRG))

# Build the vocabulary
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the model
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(1000, hid_dim)

        self.layers = nn.ModuleList(
            [TransformerBlock(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, trg, src_mask, trg_mask):
        # src = [batch size, src len]
        # trg = [batch size, trg len]
        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        batch_size = src.shape[0]
        src_len = src.shape[1]
        trg_len = trg.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(device)

        src = self.dropout((self.tok_embedding(src) * self.pos_embedding(pos) / self.scale) + self.pos_embedding(pos))

        # Add the transformer blocks
        for layer in self.layers:
            src = layer(src, src_mask)

        output = self.fc_out(src)

        return output


# Define the hyperparameters
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
N_LAYERS = 3
N_HEADS = 8
PF_DIM = 512
DROPOUT = 0.1
CLIP = 1

# Define the model
model = Transformer(INPUT_DIM, OUTPUT_DIM, HID_DIM, N_LAYERS, N_HEADS, PF_DIM, DROPOUT, device)

# Define the optimizer and criterion
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi['<pad>'])

# Move the model to the device
model = model.to(device)


# Define the training loop
def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg[:, :-1], None, None)

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [(batch size * trg len) - 1, output dim]
        # trg = [(batch size * trg len) - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# Define the evaluation loop
def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg[:, :-1], None, None)

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [(batch size * trg len) - 1, output dim]
            # trg = [(batch size * trg len) - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# Define the number of epochs and the clip value
N_EPOCHS = 10
CLIP = 1

# Continue the training process
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    print(
        f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f}')
