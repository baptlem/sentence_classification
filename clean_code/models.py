from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
from sklearn.feature_extraction.text import HashingVectorizer, FeatureHasher
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import SGDClassifier,RidgeClassifier
import warnings
from sentence_transformers import SentenceTransformer, util
from bert_score import score
from utilities import _most_common_element
from sklearn.neighbors import NearestCentroid
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.utils import shuffle

warnings.filterwarnings("ignore")


def compare_to_class(model,df_train,df_test,embedding_name):
    embedding_model = SentenceTransformer(embedding_name)
    theme = ["Politics","Health","Finance","Travel","Food","Education","Environment","Fashion","Science","Sports","Technology","Entertainment"]
    embeddings_theme = embedding_model.encode(theme)
    embeddings_train = embedding_model.encode(list(df_train['sentences']))
    embedding_test = embedding_model.encode(list(df_test['sentences']))
    results = cosine_similarity(embeddings_theme,embeddings_train).transpose()
    results_test = cosine_similarity(embeddings_theme,embedding_test).transpose()
    model.fit(results,df_train['labels'])
    predict = model.predict(results_test)
    # highest_indices = np.argsort(results)[:,-1:].flatten()
    # print(highest_indices)
    return predict

def tf_idf_classifier(df_train,df_test,classifier=SGDClassifier(),stop_words="english",train_vocab=None):
    vocab = None
    if train_vocab is not None:
        vocab = _create_vocab(train_vocab,stop_words="english")
    # print(df_train.shape)
    pipeline = Pipeline([
    ('preprocess', FunctionTransformer(_preprocess_text)),
    ('vectorizer',  TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words=stop_words,vocabulary=vocab,strip_accents= 'unicode',decode_error= 'replace')),
    ('selector',  SelectKBest(f_classif, k=500)),
    ("classifier", classifier)
    ])
    
    X_train = df_train.loc[:,"sentences"]
    y_train = df_train.loc[:,"labels"]
    # print(pipeline[:2].fit_transform(X_train,y_train).shape)
    pipeline.fit(X_train, y_train)
    predicted_label = pipeline.predict(df_test.loc[:,"sentences"])
    return predicted_label
    
def _preprocess_text(X):
    X = X.str.lower().apply(word_tokenize)  # Tokenization
    # X = X.apply(lambda x:[word for word in x if word not in stopwords.words('english')])  # Stopword Removal
    lemmatizer = WordNetLemmatizer()
    X = X.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])  # Lemmatization
    # print(X)
    return X.apply(lambda x:' '.join(x))

def _create_vocab(df_train,stop_words="english"):
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    theme = ["Politics","Health","Finance","Travel","Food","Education","Environment","Fashion","Science","Sports","Technology","Entertainment"]
    threshold =  [0.405,0.38,0.37,0.37,0.4,0.41,0.4,0.40,0.43,0.395,0.408,0.42]
    
    pipeline = Pipeline([
        
    ('preprocess', FunctionTransformer(_preprocess_text)),
    ('vectorizer',  TfidfVectorizer(sublinear_tf=True, stop_words=stop_words)),#, max_df=0.5, min_df=5
    ])
    
    predicted_label = pipeline.fit_transform(df_train.loc[:,"sentences"])
    vectorizer = pipeline.named_steps['vectorizer']
    vocabulary = vectorizer.vocabulary_
    embedding_voc= model.encode(list(vocabulary.keys()))
    embedding_theme = model.encode(theme)
    results = cosine_similarity(embedding_voc,embedding_theme)
    vectors = np.array(results) 
    # print(vectors.shape)
    # print(pd.Series(np.where(vectors > threshold)[1]).value_counts())
    # print(np.where(np.sum(vectors > threshold, axis=1) == 1)[1])
    indices = np.where(np.any(vectors > threshold, axis=1))[0]
    # print(indices)
    # indices = np.where(np.sum(vectors > threshold, axis=1) == 1)[0]
    # print(list(np.array(list(vocabulary.keys()))[indices]))
    return list(np.array(list(vocabulary.keys()))[indices])



def llm_embedding(model_name,df_train,df_test,k=25):
    # TODO replace by sklearn knn
    model = SentenceTransformer(model_name)
    model.max_seq_length = 128
    embeddings_test = model.encode(list(df_test["sentences"]))
    embedding_train = model.encode(list(df_train['sentences']))
    results = cosine_similarity(embedding_train,embeddings_test).transpose()
    highest_indices = np.argsort(results)[:,-k:]
    result = []
    for i in range(len(highest_indices)):
        highest_indices[i] = df_train.loc[highest_indices[i],'labels']
        result.append(_most_common_element(highest_indices[i]))
    return result

def llm_embedding_centroid(model_name, df_train, df_test):
    model = SentenceTransformer(model_name)
    # print(df_test)
    embeddings_test = model.encode(list(df_test["sentences"]))
    embeddings_train = model.encode(list(df_train['sentences']))

    clf = NearestCentroid()
    clf.fit(embeddings_train, df_train['labels'])

    result = clf.predict(embeddings_test)

    return result
    

 
def bert_score(df_train,df_test,k=25):
    results = []
    for sentence in df_test.loc[:,"sentences"]:
        ind_results = []
        for s_train in df_train.loc[:,"sentences"]:
            _, _, bert_score = score([s_train], [sentence], lang='en', rescale_with_baseline=True)
            ind_results.append(bert_score)
        results.append(ind_results)
        # print(bert_score)
    highest_indices = np.argsort(results)[:, :k]
    result = []
    for i in range(len(highest_indices)):
        highest_indices[i] = df_train.loc[highest_indices[i],'labels']
        result.append(_most_common_element(highest_indices[i]))
    return result
     
def softmax(x, temperature=1.0):
    e_x = np.exp(x / temperature)
    return e_x / e_x.sum(axis=1, keepdims=True)  

def finetuned_llm(model_name_llm,model_name_classifier, df_train, df_test):
    predicted_probas = []
    df_train = shuffle(df_train)
    model_llm = SentenceTransformer(model_name_llm)
    embeddings_test = model_llm.encode(list(df_test["sentences"]))
    embedding_train = model_llm.encode(list(df_train['sentences']))
    model_name_classifier.fit(embedding_train,df_train['labels'])
    
    # if isinstance(model_name_classifier,SGDClassifier):
    # predicted_probas = model_name_classifier.predict_proba(embeddings_test)
    # print(predicted_probas[0])
    # print(predicted_probas.shape)
    # predicted_probas = softmax(predicted_probas, temperature=3.0)
    # print(predicted_probas[0])
    # print(predicted_probas.shape)
    # else:
    #     predicted_probas = model_name_classifier.decision_function(embeddings_test)
    #     predicted_probas = softmax(predicted_probas, temperature=3.0)
    predicted_label = model_name_classifier.predict(embeddings_test)
    return predicted_label,predicted_probas
    



