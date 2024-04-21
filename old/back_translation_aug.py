from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import MarianMTModel, MarianTokenizer
import os
import pandas as pd
import numpy as np
import torch
import math as m
import os
print(os.path.abspath('.'))

"""
Back translation by MarianMTModel code inspired from https://amitness.com/back-translation/
"""

def translate(texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
    src_texts = [template(text) for text in texts]

    # Tokenize the texts
    encoded = tokenizer.prepare_seq2seq_batch(src_texts,
                                              return_tensors='pt')
    
    # Translation on GPU, doesn't work
    # batch_encoded = np.array_split(encoded, m.ceil(len(encoded)/64))
    # batch_translated = []
    # with torch.no_grad():
    #     for batch in batch_encoded:
    #         for k, v in batch.items():
    #             batch[k] = v.cuda()
    #         # batch = batch.cuda() #encoded.to("cuda")
    #         translated = model.generate(**batch)
    #         translated = translated.cpu() #.to('cpu')
    #         batch_translated.append(translated)
    # batch_translated = [el for batch in batch_translated for el in batch]

    translated = model.generate(**encoded)
    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    
    return translated_texts

def back_translate(texts, target_lang="fr", source_lang="en"):
    # Translate to target language
    fr_texts = translate(texts, target_model, target_tokenizer, 
                         language=target_lang)

    # Translate from target language back to source language
    back_translated_texts = translate(fr_texts, en_model, en_tokenizer, 
                                      language=source_lang)
    
    return back_translated_texts


sentences = []
labels = []
with open("nlp_kaggle/NLP_CS_kaggle/data/eda_train_set.txt", 'r') as f:
    for line in f.readlines():
        label, sentence = line.split(maxsplit=1)
        labels.append(label)
        sentences.append(sentence)


target_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
target_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
# target_tokenizer.to("cuda")
target_model = MarianMTModel.from_pretrained(target_model_name)
# target_model.to("cuda")


en_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
# en_tokenizer.to("cuda")
en_model = MarianMTModel.from_pretrained(en_model_name)
# en_model.to("cuda")


en_texts = sentences

aug_texts_es = back_translate(en_texts, source_lang="en", target_lang="es")

aug_texts_fr = back_translate(en_texts, source_lang="en", target_lang="fr")


with open("nlp_kaggle/NLP_CS_kaggle/data/backtrans_train_set.txt", 'w') as f:
    for label, en_sent, es_sent, fr_sent in zip(labels, en_texts, aug_texts_es, aug_texts_fr):
        f.write(f"{label}\t{en_sent.strip()}\n")
        f.write(f"{label}\t{es_sent.strip()}\n")
        f.write(f"{label}\t{fr_sent.strip()}\n")

        

# CroissantLLM (back translation doesn't work)
# print("Loading models...")
# model_name = "croissantllm/CroissantLLMBase"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# print("Start translation")
# inputs = tokenizer(f"I am so tired I could sleep right now. -> Je suis si fatigué que je pourrais m'endormir maintenant.\nHe is heading to the market. -> Il va au marché.\n{sentences[0]} ->", return_tensors="pt").to(model.device)
# tokens = model.generate(**inputs, max_length=100, do_sample=True, top_p=0.95, top_k=1, temperature=0.3)
# decoded = tokenizer.decode(tokens[0])
# print(decoded)
