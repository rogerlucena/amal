import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class LSTMSentiment(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, use_gpu, batch_size, dropout=0.5):
        super(LSTMSentiment, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.n_samples = 1 # necessary for "init_hidden()" so in the future we will
                           # not have an error with "forward" regarding the dimensions
                           # (default is 1, but calculated dynamically with "forward below")
        self.hidden_already_set = False

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if self.use_gpu:
            return (Variable(torch.zeros(1, self.batch_size*self.n_samples, self.hidden_dim).cuda()),
                    Variable(torch.zeros(1, self.batch_size*self.n_samples, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(1, self.batch_size*self.n_samples, self.hidden_dim)),
                    Variable(torch.zeros(1, self.batch_size*self.n_samples, self.hidden_dim)))

    def forward(self, sentence):
        x = self.embeddings(sentence).view(len(sentence), -1, self.embedding_dim) #self.batch_size, -1)
        if not self.hidden_already_set:
            self.n_samples = int(x.shape[1]/self.batch_size)
            self.hidden = self.init_hidden()
            self.hidden_already_set = True
        
        # print('x.shape:', x.shape)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs