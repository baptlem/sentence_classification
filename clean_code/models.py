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
from sklearn.linear_model import SGDClassifier
import warnings
from sentence_transformers import SentenceTransformer, util
from bert_score import score
from utilities import _most_common_element
from sklearn.neighbors import NearestCentroid

warnings.filterwarnings("ignore")



def tf_idf_classifier(df_train,df_test,classifier=SGDClassifier(),stop_words="english",train_vocab=None):
    vocab = None
    if train_vocab is not None:
        vocab = _create_vocab(train_vocab,stop_words="english")
    pipeline = Pipeline([
    ('preprocess', FunctionTransformer(_preprocess_text)),
    ('vectorizer',  TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5, stop_words=stop_words,vocabulary=vocab)),
    # ('hasher', FeatureHasher(n_features=2**10)),
    ("classifier", classifier)
    ])

    X_train = df_train.loc[:,"sentences"]
    y_train = df_train.loc[:,"labels"]
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
    threshold =  [0.45,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
    
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
    indices = np.where(np.any(vectors > threshold, axis=1))[0]
    # indices = np.where(np.sum(vectors > threshold, axis=1) == 1)[0]
    print(list(np.array(list(vocabulary.keys()))[indices]))
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
     
    

def finetuned_llm(model_name_llm,model_name_classifier, df_train, df_test):
    model_llm = SentenceTransformer(model_name_llm)
    embeddings_test = model_llm.encode(list(df_test["sentences"]))
    embedding_train = model_llm.encode(list(df_train['sentences']))
    # print(embedding_train.shape)
    model_name_classifier.fit(embedding_train,df_train['labels'])
    predicted_label = model_name_classifier.predict(embeddings_test)
    return predicted_label
    



