import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from flask import Flask, request, jsonify
import threading
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src_len, batch_size, emb_dim]
        outputs, hidden = self.rnn(embedded)
        # outputs = [src_len, batch_size, hid_dim * 2]
        # hidden = [n_layers * 2, batch_size, hid_dim]
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch_size, hid_dim]
        # encoder_outputs = [src_len, batch_size, hid_dim * 2]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat hidden src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden = [batch_size, src_len, hid_dim]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch_size, src_len, hid_dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch_size, src_len, hid_dim]

        attention = self.v(energy).squeeze(2)
        # attention = [batch_size, src_len]

        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((hid_dim * 2) + emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch_size]
        # hidden = [n_layers, batch_size, hid_dim]
        # encoder_outputs = [src_len, batch_size, hid_dim * 2]
        input = input.unsqueeze(0)
        # input = [1, batch_size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch_size, emb_dim]

        a = self.attention(hidden[-1], encoder_outputs)
        # a = [batch_size, src_len]

        a = a.unsqueeze(1)
        # a = [batch_size, 1, src_len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch_size, src_len, hid_dim * 2]

        weighted = torch.bmm(a, encoder_outputs)
        # weighted = [batch_size, 1, hid_dim * 2]

        weighted = weighted.permute(1, 0, 2)
        # weighted = [1, batch_size, hid_dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input = [1, batch_size, (hid_dim * 2) + emb_dim]

        output, hidden = self.rnn(rnn_input, hidden)
        # output = [1, batch_size, hid_dim]
        # hidden = [n_layers, batch_size, hid_dim]

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # prediction = [batch_size, output_dim]

        return prediction, hidden, a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        # hidden = [n_layers * 2, batch_size, hid_dim] -> [n_layers, batch_size, hid_dim * 2]
        hidden = hidden.view(self.encoder.rnn.num_layers, 2, batch_size, -1)
        hidden = torch.sum(hidden, dim=1)  # sum bidirectional outputs

        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs

# Simple vocabulary class
class Vocab:
    def __init__(self):
        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.freqs = {}

    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def __len__(self):
        return len(self.word2idx)

# TranslationDataset
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, src_vocab, trg_vocab, max_len=50):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src = self.src_sentences[idx]
        trg = self.trg_sentences[idx]

        src_indices = [self.src_vocab.word2idx.get(w, self.src_vocab.word2idx['<unk>']) for w in src]
        trg_indices = [self.trg_vocab.word2idx.get(w, self.trg_vocab.word2idx['<unk>']) for w in trg]

        src_indices = [self.src_vocab.word2idx['<sos>']] + src_indices + [self.src_vocab.word2idx['<eos>']]
        trg_indices = [self.trg_vocab.word2idx['<sos>']] + trg_indices + [self.trg_vocab.word2idx['<eos>']]

        src_tensor = torch.tensor(src_indices[:self.max_len], dtype=torch.long)
        trg_tensor = torch.tensor(trg_indices[:self.max_len], dtype=torch.long)

        return src_tensor, trg_tensor

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=0, batch_first=False)
    trg_batch = nn.utils.rnn.pad_sequence(trg_batch, padding_value=0, batch_first=False)
    return src_batch, trg_batch

# Training function
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for src, trg in iterator:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Evaluation function
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, trg in iterator:
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0)  # turn off teacher forcing

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Translation function
def translate_sentence(model, sentence, src_vocab, trg_vocab, device, max_len=50):
    model.eval()

    tokens = sentence.lower().split()

    src_indices = [src_vocab.word2idx.get(token, src_vocab.word2idx['<unk>']) for token in tokens]
    src_indices = [src_vocab.word2idx['<sos>']] + src_indices + [src_vocab.word2idx['<eos>']]
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(1).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

        hidden = hidden.view(model.encoder.rnn.num_layers, 2, 1, -1)
        hidden = torch.sum(hidden, dim=1)

    trg_indices = [trg_vocab.word2idx['<sos>']]

    for i in range(max_len):
        trg_tensor = torch.tensor([trg_indices[-1]], dtype=torch.long).to(device)

        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs)

        pred_token = output.argmax(1).item()

        trg_indices.append(pred_token)

        if pred_token == trg_vocab.word2idx['<eos>']:
            break

    trg_tokens = []
    for i in trg_indices:
        if i in trg_vocab.idx2word:
            trg_tokens.append(trg_vocab.idx2word[i])
        else:
            trg_tokens.append('<unk>')

    return trg_tokens[1:-1]  # remove <sos> and <eos>

# Main training and setup
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    INPUT_DIM = 1000  # vocabulary size
    OUTPUT_DIM = 1000
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    # Create vocabularies (simplified)
    src_vocab = Vocab()
    trg_vocab = Vocab()

    # Add some sample words
    sample_words = ['hello', 'world', 'how', 'are', 'you', 'i', 'am', 'fine', 'thank', 'good', 'morning', 'afternoon', 'evening']
    for word in sample_words:
        src_vocab.add_word(word)
        trg_vocab.add_word(word)

    # Create model
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    attention = Attention(HID_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attention)
    model = Seq2Seq(enc, dec, device).to(device)

    # Create sample data
    src_sentences = [['hello', 'world'], ['how', 'are', 'you'], ['good', 'morning']]
    trg_sentences = [['bonjour', 'monde'], ['comment', 'allez', 'vous'], ['bonjour']]

    # Add French words to vocab
    french_words = ['bonjour', 'monde', 'comment', 'allez', 'vous']
    for word in french_words:
        trg_vocab.add_word(word)

    dataset = TranslationDataset(src_sentences, trg_sentences, src_vocab, trg_vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # Training
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    N_EPOCHS = 10
    CLIP = 1

    for epoch in range(N_EPOCHS):
        train_loss = train(model, dataloader, optimizer, criterion, CLIP)
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f}')

    # Test translation
    test_sentence = "hello world"
    translation = translate_sentence(model, test_sentence, src_vocab, trg_vocab, device)
    print(f'Input: {test_sentence}')
    print(f'Translation: {" ".join(translation)}')

    # Save model
    torch.save(model.state_dict(), 'nmt_model.pth')
    print("Model saved as nmt_model.pth")

# Flask API for serving translations
app = Flask(__name__)

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    sentence = data.get('sentence', '')
    if not sentence:
        return jsonify({'error': 'No sentence provided'}), 400

    try:
        translation = translate_sentence(model, sentence, src_vocab, trg_vocab, device)
        return jsonify({'translation': ' '.join(translation)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

def run_server():
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    INPUT_DIM = 1000  # vocabulary size
    OUTPUT_DIM = 1000
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    # Create vocabularies (simplified)
    src_vocab = Vocab()
    trg_vocab = Vocab()

    # Add some sample words
    sample_words = ['hello', 'world', 'how', 'are', 'you', 'i', 'am', 'fine', 'thank', 'good', 'morning', 'afternoon', 'evening']
    for word in sample_words:
        src_vocab.add_word(word)
        trg_vocab.add_word(word)

    # Create model
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    attention = Attention(HID_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attention)
    model = Seq2Seq(enc, dec, device).to(device)

    # Load model if exists, otherwise train
    try:
        model.load_state_dict(torch.load('nmt_model.pth', map_location=device))
        print("Model loaded from nmt_model.pth")
    except FileNotFoundError:
        print("Model not found, training new model...")
        # Create sample data
        src_sentences = [['hello', 'world'], ['how', 'are', 'you'], ['good', 'morning']]
        trg_sentences = [['bonjour', 'monde'], ['comment', 'allez', 'vous'], ['bonjour']]

        # Add French words to vocab
        french_words = ['bonjour', 'monde', 'comment', 'allez', 'vous']
        for word in french_words:
            trg_vocab.add_word(word)

        dataset = TranslationDataset(src_sentences, trg_sentences, src_vocab, trg_vocab)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

        # Training
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        N_EPOCHS = 10
        CLIP = 1

        for epoch in range(N_EPOCHS):
            train_loss = train(model, dataloader, optimizer, criterion, CLIP)
            print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f}')

        # Save model
        torch.save(model.state_dict(), 'nmt_model.pth')
        print("Model saved as nmt_model.pth")

    # Test translation
    test_sentence = "hello world"
    translation = translate_sentence(model, test_sentence, src_vocab, trg_vocab, device)
    print(f'Input: {test_sentence}')
    print(f'Translation: {" ".join(translation)}')

    # Start Flask server in a separate thread
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    print("Flask server started on port 5000")

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")