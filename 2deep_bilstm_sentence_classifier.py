# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
torch.set_num_threads(8)
import sys
import codecs
import random
import torch.utils.data as Data


SEED = 1
random.seed(SEED)


# input: a sequence of tokens, and a token_to_index dictionary
# output: a LongTensor variable to encode the sequence of idxs
def prepare_sequence(seq, to_ix, cuda=False):
    var = autograd.Variable(torch.LongTensor([to_ix[w] for w in seq.split(' ')]))
    return var

def prepare_label(label,label_to_ix, cuda=False):
    var = autograd.Variable(torch.LongTensor([label_to_ix[label]]))
    return var

def build_token_to_ix(sentences):
    token_to_ix = dict()
    print(len(sentences))
    for sent in sentences:
        for token in sent.split(' '):
            if token not in token_to_ix:
                token_to_ix[token] = len(token_to_ix)
    token_to_ix['<pad>'] = len(token_to_ix)
    return token_to_ix

def build_label_to_ix(labels):
    label_to_ix = dict()
    for label in labels:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)


def get_all_files_from_dir(dirpath):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(dirpath) if isfile(join(dirpath, f))]
    return onlyfiles

def head_n(fname, n=2):
    count = 1
    print ('---------------------------------------')
    f = open(fname)
    for line in f:
        print (line)
        count += 1
        if count > n:
            print ('-----------------------------------')
            f.close()
            return
#head_n(test_file_pos,10)
#thos function would only extract vp and vn files, in order to extract vpn = vp+vn in one file, give last arguement as False/0 

def extract_names(l_files,patt,p_xor_n = True):
    # note that you may need to 
    r = []
    for f in l_files:
        tokens = f.split(".")
        if tokens[-3]==patt:
            if p_xor_n:
                if tokens[-2] != 'vpn':
                    r.append(f)
                    #yield f
            elif tokens[-2] == 'vpn':
                return f
    return sorted(r)
#extract_names(files,'test')
def get_sentence_out(path):
    f = open(path)
    return map(lambda x:x.split(",")[2],f)
def get_neg_pos_sent(type_,files,path):
    return [get_sentence_out(path+n) for n in extract_names(files,type_)]
def load_stanford_data():
    fpath = './cross_validation_data/vpn_filtered/'
    files = get_all_files_from_dir(fpath)
    
    train_sent_neg,train_sent_pos = get_neg_pos_sent('train',files,fpath)
    val_sent_neg,val_sent_pos = get_neg_pos_sent('dev',files,fpath)
    test_sent_neg,test_sent_pos = get_neg_pos_sent('test',files,fpath)
    
    train_data = [(sent,1) for sent in train_sent_pos] + [(sent, 0) for sent in train_sent_neg]
    dev_data = [(sent, 1) for sent in val_sent_pos] + [(sent, 0) for sent in val_sent_neg]
    test_data = [(sent, 1) for sent in test_sent_pos] + [(sent, 0) for sent in test_sent_neg]
    
    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)

    print('train:',len(train_data),'dev:',len(dev_data),'test:',len(test_data))
    
    word_to_ix = build_token_to_ix([s for s,_ in train_data+dev_data+test_data])
    label_to_ix = {0:0,1:1}
    print('vocab size:',len(word_to_ix),'label size:',len(label_to_ix))
    print('loading data done!')
    return train_data,dev_data,test_data,word_to_ix,label_to_ix

# Load the data and define hyperparametars and specifications ###########################
train_data, dev_data, test_data, word_to_ix, label_to_ix = load_stanford_data()
EMBEDDING_DIM = 100
HIDDEN_DIM = 100
EPOCH = 20
NUM_LAYERS = 2
loss_function = nn.NLLLoss()#negative log likelihood loss 
########################################################################################


#Define the BiLSTM class################################################################

class BiLSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_layers, vocab_size, label_size):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,num_layers,bidirectional=True)
        self.hidden2label = nn.Linear(2*hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(self.num_layers*2, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(self.num_layers*2, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs
########################################################################################

#Define the helper functions for the accuracy###########################################
def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     right = 0
     for i in range(len(truth)):
         if truth[i]==pred[i]:
             right += 1.0
     return right/len(truth)

def evaluate(model, data, loss_function, word_to_ix, label_to_ix, name ='dev'):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []

    for sent, label in data:
        truth_res.append(label_to_ix[label])
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        sent = prepare_sequence(sent, word_to_ix)
        label = prepare_label(label, label_to_ix)
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res.append(pred_label)
        # model.zero_grad() # should I keep this when I am evaluating the model?
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ' avg_loss:%g train acc:%g' % (avg_loss, acc ))
    return acc
########################################################################################


#Define training per epoch###########################################################

def train_epoch(model, train_data, loss_function, optimizer, word_to_ix, label_to_ix, i):
    model.train()
    
    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    batch_sent = []

    for sent, label in train_data:


        truth_res.append(label_to_ix[label])
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        sent = prepare_sequence(sent, word_to_ix)
        label = prepare_label(label, label_to_ix)
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res.append(pred_label)
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
        count += 1
        if count % 500 == 0:
            print('epoch: %d iterations: %d loss :%g' % (i, count, loss.data[0]))

        loss.backward()
        optimizer.step()
    avg_loss /= len(train_data)
    print('epoch: %d done! \n train avg_loss:%g , acc:%g'%(i, avg_loss, get_accuracy(truth_res,pred_res)))
########################################################################################



# The main function that you call for training #########################################
def train():
    model = BiLSTMClassifier(embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM, num_layers = NUM_LAYERS,
                           vocab_size=len(word_to_ix),label_size=len(label_to_ix))
    best_dev_acc = 0.0
    loss_function = nn.NLLLoss()#negative log likelihood loss 
    optimizer = optim.Adam(model.parameters(),lr = 1e-3)
    #optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)
    f = open('train_test.txt','w')
    no_up = 0
    for i in range(EPOCH):
        random.shuffle(train_data)
        print('epoch: %d start!' % i)
        train_epoch(model, train_data, loss_function, optimizer, word_to_ix, label_to_ix, i)
        print('now best dev acc:',best_dev_acc)
        dev_acc = evaluate(model,dev_data,loss_function,word_to_ix,label_to_ix,'dev')
        #dev_accs.append(dev_acc)
        test_acc = evaluate(model, test_data, loss_function, word_to_ix, label_to_ix, 'test')
        #test_accs.append(test_acc)
        f.write(str(dev_acc)+","+str(test_acc))
        torch.save(model.state_dict(), 'best_models/2deep_bilstm_best_model_epoch_'+str(i) + '.model')
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            print('New Best Dev!!!')
            
            no_up = 0
        else:
            no_up += 1
            if no_up >= 10:
                f.close()
                return model, i 
                exit()
    f.close()
    return model,i
#train~######
model, i = train()
#torch.save(model.state_dict(), 'best_models/bilstm_best_model_Epoch_'+str(i)+'_acc_' + str(int(test_acc*10000)) + '.model')
