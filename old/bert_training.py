import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import BertModel, BertForSequenceClassification
from transformers import BertTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch import nn
# from torch.optim import AdamW
from transformers import AdamW
from pytorch_transformers.optimization import WarmupLinearSchedule
from peft import LoraConfig, TaskType
from peft import get_peft_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.model_selection import train_test_split
import inspect 
import collections
from clean_code.utilities import import_data 
import os
print(os.path.abspath('.'))

"""
Code inspired from https://medium.com/@coderhack.com/fine-tuning-bert-for-text-classification-a-step-by-step-guide-1a1c5f8e8ae1
https://github.com/nlptown/nlp-notebooks/blob/master/Text%20classification%20with%20BERT%20in%20PyTorch.ipynb
https://medium.com/@karkar.nizar/fine-tuning-bert-for-text-classification-with-lora-f12af7fa95e4
"""

MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
EPOCH = 50
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-5 #32
# LEARNING_RATE = 5e-5 #16
# LEARNING_RATE = 1e-6 #8
WARMUP_PROPORTION = 0.1
MAX_GRAD_NORM = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

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

class BertInputItem(object):
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask, segment_ids, label_id):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        
def convert_examples_to_inputs(example_texts, example_labels, label2idx, max_seq_length, tokenizer, verbose=0):
    """Loads a data file into a list of `InputBatch`s."""
    
    input_items = []
    examples = zip(example_texts, example_labels)
    for (ex_index, (text, label)) in enumerate(examples):

        # Create a list of token ids
        input_ids = tokenizer.encode(f"[CLS] {text} [SEP]")
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]

        # All our tokens are in the first input segment (id 0).
        segment_ids = [0] * len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # label_id = label2idx[label]

        input_items.append(
            BertInputItem(text=text,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label))

        
    return input_items

def get_data_loader(features, max_seq_length, batch_size, shuffle=True): 

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return dataloader


df_train = pd.read_json("./NLP/kaggle/NLP_CS_kaggle/data/train.json")
label2idx = {col:i for i,col in enumerate(df_train.columns)}

tokenizer = BertTokenizer.from_pretrained(#'google-bert/bert-base-uncased',
                                          'distilbert/distilbert-base-uncased'
                                          # do_lower_case=True 
                                          )  

model = BertForSequenceClassification.from_pretrained(#'google-bert/bert-base-uncased'
                                                       'distilbert/distilbert-base-uncased',
                                                        num_labels=len(label2idx)
                                                    ).cuda()


# lora_config = LoraConfig(
#     task_type=TaskType.SEQ_CLS, r=1, lora_alpha=1, lora_dropout=0.1
# )
# model = get_peft_model(model, lora_config)

#Fine-tuning only the classifier
for param in model.bert.parameters():
    param.requires_grad = False

# model recovery
# checkpoint = torch.load("./NLP/kaggle/NLP_CS_kaggle/finetuned_bert/best_distilbert_0804_lora.bin", map_location=device)
# model.load_state_dict(checkpoint)



# sentences = []
# labels = []
# with open("./NLP/kaggle/NLP_CS_kaggle/data/free_augmentation_100_0904.txt", 'r') as f:
#     for line in f.readlines():
#         label, sentence = line.split(maxsplit=1)
#         labels.append(label)
#         sentences.append(sentence.strip())
# train_set = np.array([labels, sentences]).T

# sentences = []
# labels = []
# with open("./NLP/kaggle/NLP_CS_kaggle/data/annotated_test.txt", 'r') as f:
#     for line in f.readlines():
#         label, sentence = line.split(maxsplit=1)
#         labels.append(label)
#         sentences.append(sentence.strip())
# print(len(sentences))
# test_set = np.array([labels, sentences]).T

train, test = import_data("./NLP/kaggle/NLP_CS_kaggle/data/concat_dataset.txt", "./NLP/kaggle/NLP_CS_kaggle/data/annotated_test.txt")

train_set = train.to_numpy()
test_set = test.to_numpy()


X_train, X_val, y_train, y_val = train_set[:,1], test_set[:,1], train_set[:,0], test_set[:,0]
# X_train, X_val, y_train, y_val = train_test_split(train_set[:,1], train_set[:,0], test_size=0.1)
y_train = y_train.astype(np.uint8)
y_val = y_val.astype(np.uint8)

train_features = convert_examples_to_inputs(X_train, y_train, label2idx, MAX_SEQ_LENGTH, tokenizer, verbose=0)
val_features = convert_examples_to_inputs(X_val, y_val, label2idx, MAX_SEQ_LENGTH, tokenizer)

train_dataloader = get_data_loader(train_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=True)
val_dataloader = get_data_loader(val_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=True)

def evaluate(model, dataloader):
    model.eval()
    
    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []

    for step, batch in enumerate(tqdm(dataloader, desc="Evaluation iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            outputs_mod = model(input_ids, attention_mask=input_mask,
                                          token_type_ids=segment_ids, labels=label_ids)

        outputs = np.argmax(outputs_mod.logits.to('cpu'), axis=1)
        label_ids = label_ids.to('cpu').numpy()
        
        predicted_labels += list(outputs)
        correct_labels += list(label_ids)
        
        eval_loss += outputs_mod.loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    
    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)
        
    return eval_loss, correct_labels, predicted_labels

num_train_steps = int(len(train_dataloader.dataset) / BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS * EPOCH)
num_warmup_steps = int(WARMUP_PROPORTION * num_train_steps)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, correct_bias=False)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps)

OUTPUT_DIR = "./NLP/kaggle/NLP_CS_kaggle/finetuned_bert"
MODEL_FILE_NAME = "distilbert_ds_2004_concat.bin"
PATIENCE = 5
device = "cuda"
loss_history = []
no_improvement = 0
for _ in trange(int(EPOCH), desc="Epoch"):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Training iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
        loss = outputs[0]

        if GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        loss.backward()
        tr_loss += loss.item()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)  
            
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
    dev_loss, _, _ = evaluate(model, val_dataloader)
    
    print("Loss history:", loss_history)
    print("Dev loss:", dev_loss)
    
    if len(loss_history) == 0 or dev_loss < min(loss_history):
        no_improvement = 0
        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = os.path.join(OUTPUT_DIR, "best_"+MODEL_FILE_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
    else:
        no_improvement += 1

    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)

    if no_improvement >= PATIENCE: 
        print("No improvement on development set. Finish training.")
        break
        
    
    loss_history.append(dev_loss)


plt.plot(np.arange(len(loss_history)), loss_history)
plt.title("Loss over epoch")
plt.savefig("./NLP/kaggle/NLP_CS_kaggle/distilbert_ds_2004_concat.png")
