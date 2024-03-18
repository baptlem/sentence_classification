import os
import pandas as pd
import numpy as np
import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import BertModel
from transformers import BertTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import os
print(os.path.abspath('.'))


"""
Code inspired from https://medium.com/@coderhack.com/fine-tuning-bert-for-text-classification-a-step-by-step-guide-1a1c5f8e8ae1
"""

BATCH_SIZE = 32
EPOCH = 50

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, sentences, labels):
        'Initialization'
        self.labels = labels
        self.sentences = sentences

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        x = self.sentences[index]
        y = self.labels[index]

        return x, y


def train(model, optimizer, train_loader, criterion, start_i=0):
    model.train()
    total_loss = 0
    i=start_i
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch 
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        writer.add_scalar("Loss/train", loss, i)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        i += 1
    print(f'Training loss: {total_loss/len(train_loader)}')


def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch  
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            total_acc += (predictions == labels).sum().item()

    print(f'Test loss: {total_loss/len(test_loader)} Test acc: {total_acc/len(y_val)*100}%')



sentences = []
labels = []
with open("./kaggle/NLP_CS_kaggle/data/backtrans_eda_train_set.txt", 'r') as f:
    for line in f.readlines():
        label, sentence = line.split(maxsplit=1)
        labels.append(label)
        sentences.append(sentence.strip())

tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased',
                                          # do_lower_case=True 
                                          )  
model = BertModel.from_pretrained('google-bert/bert-base-uncased')
tokens_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])).cuda()

pred = model.forward(tokens_ids)
print(pred)

raise

a = np.array([sentences, labels]).T
X_train, X_val, y_train, y_val = train_test_split(a[:,0], a[:,1], test_size=0.05, shuffle=True)


print(a.shape)
print(len(X_train), len(y_train))
print(len(X_val), len(y_val))


writer = SummaryWriter()
model = BertModel.from_pretrained('google-bert/bert-base-uncased')
classifier = nn.Linear(768, 12)
model = nn.Sequential(model, classifier)
criterion = nn.CrossEntropyLoss()  
optimizer = AdamW(model.parameters(), lr=2e-5)

# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
params = {'batch_size': BATCH_SIZE,
          'shuffle': True,
          'num_workers': 0}

# Generators
training_set = Dataset(X_train, y_train)
training_generator = DataLoader(training_set, **params)

validation_set = Dataset(X_val, y_val)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)


i = 0
for epoch in range(EPOCH):
    for local_batch, local_labels in training_generator:
        print(local_batch)
        print(local_labels)
        raise
    train(model, optimizer, training_generator, criterion, start_i=i)
    evaluate(model, validation_generator, criterion)
    i += BATCH_SIZE

writer.flush()
torch.save(model.state_dict(), 'fine_tuned_bert.pt')

writer.close()