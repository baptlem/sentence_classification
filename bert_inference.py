from transformers import BertForSequenceClassification, BertConfig
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from peft import LoraConfig, TaskType
from sklearn.metrics import accuracy_score
from peft import get_peft_model
from tqdm import tqdm
from clean_code.utilities import evaluation, import_data
import numpy as np
import pandas as pd
import torch
import os
print(os.path.abspath('.'))

MAX_SEQ_LENGTH=512
BATCH_SIZE = 2
device = "cuda" if torch.cuda.is_available() else "cpu"

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


tokenizer = BertTokenizer.from_pretrained(#'google-bert/bert-base-uncased',
                                          'distilbert/distilbert-base-uncased',
                                          # do_lower_case=True 
                                          )  

checkpoint = torch.load("./NLP/kaggle/NLP_CS_kaggle/finetuned_bert/best_distilbert_ds_2004_concat.bin", map_location=torch.device('cpu'))


bert_config = BertConfig(num_labels=12)
model = BertForSequenceClassification(bert_config)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=1, lora_alpha=1, lora_dropout=0.1
)
# model = get_peft_model(model, lora_config)
model.to(device)

model.load_state_dict(checkpoint)


df_train = pd.read_json("./NLP/kaggle/NLP_CS_kaggle/data/train.json")
label2idx = {col:i for i,col in enumerate(df_train.columns)}

true_labels = []
sentences = []
with open("./NLP/kaggle/NLP_CS_kaggle/data/annotated_test.txt", 'r') as f:
    for line in f.readlines():
        label, sentence = line.split(maxsplit=1)
        true_labels.append(int(label))
        sentences.append(sentence)

true_labels = np.array(true_labels)



train_features = convert_examples_to_inputs(sentences, true_labels, label2idx, MAX_SEQ_LENGTH, tokenizer, verbose=0) 
train_dataloader = get_data_loader(train_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=False)

predict_labels = []
for  batch in tqdm(train_dataloader, desc="Evaluation iteration"):
    batch = tuple(t.to(device) for t in batch)
    input_ids, input_mask, segment_ids, label_ids = batch

    outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
    logits = F.softmax(outputs.logits)
    for l in logits:
        predict_label = torch.argmax(l).cpu().numpy()
        predict_labels.append(predict_label.item())
    # print(f"True label: {label_ids}, predict label: {torch.argmax(logits)}")


accuracy = accuracy_score(true_labels, np.array(predict_labels))
print(f"Accuracy of DistilBert: {accuracy}")
predict_labels = [predict_labels]

train, test = import_data("./NLP/kaggle/NLP_CS_kaggle/data/concat_dataset.txt", "./NLP/kaggle/NLP_CS_kaggle/data/annotated_test.txt")

evaluation(predict_labels, test)
