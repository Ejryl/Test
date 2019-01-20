import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from collections import Counter
import collections
from rnn import RNNModel
import itertools
from lstm import CharLSTM
def get_batches(arr, n_seqs_in_a_batch, n_characters):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''

    batch_size = n_seqs_in_a_batch * n_characters
    n_batches = len(arr) // batch_size

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size]
    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs_in_a_batch, -1))
    for n in range(0, arr.shape[1], n_characters):
        # The features
        x = arr[:, n:n + n_characters]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + n_characters]
            yield x, y
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]

EOS = '<eos>'

def read_file(filename):
    tokens = []
    with open(filename, encoding='utf8') as f:
        for line in f:
            tokens += line.split() + [EOS]
    return np.array(tokens)
start_time = time.time()

print("Reading file ... (1/5)")
trn_tok = read_file("./wiki.train.tokens")[:int(2*1e6)]
print("File read in " + str(time.time()-start_time) + " s")
print("Concatenating tokens ... (2/5)")


# trn_tok = np.concatenate(trn_tok)[:int(1e5)]

# cnt = Counter(word for sent in trn_tok for word in sent)
#
# itos = [o for o,c in cnt.most_common()]
# itos.insert(0,'<pad>')
#
# stoi = collections.defaultdict(lambda : 5, {w:i for i,w in enumerate(itos)})
#
# train_ids = np.array([([stoi[w] for w in s]) for s in trn_tok])
print("Token creation ... (3/5)")
characters = sorted(list(set(trn_tok)))

print("Dictionaries creation ... (4/5)")
int2char = dict(enumerate(characters))
char2int = {char: index for index, char in int2char.items()}
print("Generating encoded data ... (5/5)")
encoded = np.array([char2int[char] for char in trn_tok])

# encoded = trn_tok
print("Data loaded")

vocab_size = len(char2int)
hs = 1150
n_fac = 400
sequence_len = 70
batch_size = 30
#0.25, 0.1, 0.2, 0.02, 0.15
# net = CharLSTM(sequence_len=sequence_len, vocab_size=vocab_size, hidden_dim=hs, batch_size=batch_size, n_fac=n_fac, device="cuda:0")
net = RNNModel(rnn_type="LSTM", ntoken=vocab_size, ninp=hs, nhid=hs, nlayers=3, dropout=0.25, dropouth=0.1, dropouti=0.2, dropoute=0.02, wdrop=0,tie_weights=False, device="cuda:0")
try:
    net.to(net.device)
except:
    net.to(net.device)

# optimizer = optim.Adam(net.parameters(), lr=30, weight_decay=0.0001 )
optimizer = torch.optim.SGD(net.parameters(), lr=1e3, momentum=0.90, weight_decay=1.2e-6, nesterov=False)
# get the validation and the training data
val_idx = int(len(encoded) * (1 - 0.1))
data, val_data = encoded[:val_idx], encoded[val_idx:]

# empty list for the validation losses
val_losses = list()
samples = list()
print()
print("start")
cutoffs = []
cutoffs = [round(vocab_size/15), 3*round(vocab_size/15)]
# criterion = nn.AdaptiveLogSoftmaxWithLoss(in_features=hs,
#                                       n_classes=vocab_size,
#                                       cutoffs=cutoffs,
#                                       div_value=4).cuda()
criterion = nn.CrossEntropyLoss()


for epoch in range(40):

    net.init_hidden()
    start_time = time.time()
    loss = 0.
    j = 0
    net.train()
    #hidden = net.init_hidden()
    for i, (x, y) in enumerate(get_batches(data, sequence_len, batch_size)):
        x_train = torch.from_numpy(x).type(torch.LongTensor).to(net.device)
        targets = torch.from_numpy(y.T).type(torch.LongTensor).to(net.device)  # tensor of the target
        optimizer.zero_grad()
        output = net(x_train)
        del x_train

        batch_loss = criterion(output, targets.contiguous().view(-1))
        del targets
        batch_loss.backward()
        loss += float(batch_loss.item())
        optimizer.step()
        del batch_loss

        j += 1

    loss = loss/j

    net.init_hidden()
    net.eval()
    val_loss = 0.
    j = 0
    for val_x, val_y in get_batches(val_data, sequence_len, batch_size):
        val_x = torch.from_numpy(val_x).type(torch.LongTensor).to(net.device)
        val_y = torch.from_numpy(val_y.T).type(torch.LongTensor).contiguous().view(-1).to(net.device)
        val_output = net(val_x)
        #val_output = net(val_x)
        batch_val_loss = float(criterion(val_output, val_y.contiguous().view(-1)).item())
        val_loss += batch_val_loss
        del batch_val_loss

        j += 1
    val_loss = val_loss/j

    print("Epoch: {}, Batch: {}, Train Loss: {:.6f}, Validation Loss: {:.6f}, Elasped time : {:.1f}s".format(epoch, i, loss,
                                                                                             val_loss, time.time()-start_time))

net.device = "cpu"
net = net.to(net.device)
def print_text(seq):
    print(''.join([int2char[x] + " " for x in seq]))
print_text(net.predict([0], 3, 2048))
