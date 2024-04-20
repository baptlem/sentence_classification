import os
print(os.path.abspath('.'))
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
import torch
import accelerate
import re

df_train = pd.read_json("./NLP/kaggle/NLP_CS_kaggle/data/train.json")
label2idx = {col:i for i,col in enumerate(df_train.columns)}
idx2label = {str(i):col for i,col in enumerate(df_train.columns)}

sentences = []
labels = []
with open("./NLP/kaggle/NLP_CS_kaggle/data/wrong_pred_2.txt", 'r') as f:
    for line in f.readlines()[1:]:
        label, sentence = line.split(maxsplit=1)
        labels.append(label)
        sentences.append(sentence.strip())


print('Model loading...')
model = "meta-llama/Llama-2-13b-chat-hf"
# model = "mistralai/Mistral-7B-v0.1"


# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")


# tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("Model loaded !")


parameters = {
        # "max_length": 512,
        "max_new_tokens": 75,
        "top_p": 0.9,
        "return_full_text": False,
        "do_sample": True,
        "num_return_sequences": 1,
        "temperature": 1.4,
        "repetition_penalty": 1.15,
        "length_penalty": 1.0,
    }


# def prompt_mistral(prompt, label):
#     return f"<s>[INST] Change the words in the following text while keeping the topic education, don't explain how you did it and with only the output and no other phrases: The benefits of a service-learning education are many, including improved civic engagement and empathy skills. [/INST] A service-learning educational approach has numerous advantages, such as enhanced civic involvement and empathetic abilities.</s>[INST] Change the words in the following text while keeping the topic {label}, don't explain how you did it and with only the output and no other phrases: {prompt} [/INST]"


# def prompt_mistral(prompt, label):
#     return f"<s>[INST] Generate a small text following the topic of {label}. Don't be too long and only give me the answer without any explanation of how you did it. [/INST] {prompt} </s>[INST]  Generate a small text following the topic of {label}. Don't be too long and only give me the answer without any explanation of how you did it. [/INST]"


def prompt_class_init(prompt, label):
    p = f'<s>[INST]I want you to write a general statement on the theme "{label}". Find a balance between exploration and specialization. Start the sentence with "The". The sentence cant ever exceed 21 words long.[/INST] {prompt}\
        </s>[INST]I want you to write a general statement on the theme "{label}". Find a balance between exploration and specialization. Start the sentence with "The". The sentence cant ever exceed 21 words long.[/INST]'
    return p

def prompt_class(prompt, label, next_prompt):
    p = f'<s>[INST]I want you to write a general statement on the theme "{label}". Find a balance between exploration and specialization. Start the sentence with "The". The sentence cant ever exceed 21 words long.[/INST] {prompt}\
        </s>[INST]I want you to write a general statement on the theme "{label}". Find a balance between exploration and specialization. Start the sentence with "The". The sentence cant ever exceed 21 words long.[/INST] {next_prompt}\
        </s>[INST]I want you to write a general statement on the theme "{label}". Find a balance between exploration and specialization. Start the sentence with "The". The sentence cant ever exceed 21 words long.[/INST]'
    return p



# Llama2
print("writing")
f = open("./NLP/kaggle/NLP_CS_kaggle/data/corrected_pred.txt", 'w')
f.close()

with tqdm(total=len(labels)) as pbar:
    for label, sent in zip(labels, sentences):
        # aug_sentences = [sent.strip()]
        aug_sentences = []
        curr_sent = sent.strip()
        input_message = prompt_class_init(curr_sent, idx2label[str(label)])
        
        for _ in tqdm(range(25)):
            
            sequences = pipeline(input_message,
            **parameters)
            for seq in sequences:
                s = seq['generated_text']
            # print(s)
            curr_sent = s.replace("\n\n", "\n")
            curr_sent = curr_sent.split('\n')
            if len(curr_sent) == 1:
                curr_sent = curr_sent[0]
            else:
                curr_sent = curr_sent[1]

            split_curr = curr_sent.split(".")
            if split_curr[-1] != '' and len(split_curr)>1:
                split_curr.pop(-1)
                curr_sent = ''.join(split_curr)+'.'
            curr_sent = curr_sent.split(":")[-1].strip()
            curr_sent = curr_sent.strip().replace('"', '')
            aug_sentences.append(curr_sent)
            input_messag = prompt_class(aug_sentences[0], idx2label[str(label)], curr_sent)

        with open("./NLP/kaggle/NLP_CS_kaggle/data/corrected_pred.txt", 'a') as f:
            for s in aug_sentences:
                print(s)
                try:
                    f.write(f"{label}\t{s}\n")
                except:
                    continue
        
        pbar.update(1)
        continue

