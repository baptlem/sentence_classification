from utilities import most_common_element,evaluation,import_data,aggregate_voter,save_prediction,load_prediction
from models import tf_idf_classifier,llm_embedding,finetuned_llm,bert_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    
    # TODO data quality, feature hashing,other embeddings,function to save predictions,lstm?
    
    df_train,df_test = import_data("data/train_set_3103.txt","data/annotated_test.txt")
    
    sgd_classifier = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42)
    liste_of_classifier = [sgd_classifier,NearestCentroid()
                           #,ComplementNB(),KNeighborsClassifier(),NearestCentroid(),LinearSVC()
                           ]
    liste_of_llm_embeddings = ['sentence-transformers/all-MiniLM-L6-v2','bert-base-nli-mean-tokens']
    list_of_finetuned_llm = []
    
    score = [bert_score(df_train,df_test)]
    
    liste_of_preds_classif = []
    for model in tqdm(liste_of_classifier):
        predicted = tf_idf_classifier(df_train,df_test,classifier=model)
        liste_of_preds_classif.append(predicted)
    evaluation(liste_of_preds_classif,df_test)
    # save_prediction(liste_of_preds_classif,"classif")
    
    liste_of_preds_llm_embeddings = []
    for model in tqdm(liste_of_llm_embeddings):
        pass
    #     predicted = llm_embedding(model,df_train,df_test)
    #     liste_of_preds_llm_embeddings.append(predicted)
    # save_prediction(liste_of_preds_llm_embeddings,"llm_embeddings")
    liste_of_preds_llm_embeddings = load_prediction("llm_embeddings")
    evaluation(liste_of_preds_llm_embeddings,df_test)

    liste_of_preds_llm_finetuned = []
    for model in tqdm(list_of_finetuned_llm):
        predicted = finetuned_llm(model,df_train,df_test)
        liste_of_preds_llm_finetuned.append(predicted)
    evaluation(liste_of_preds_llm_finetuned,df_test)
    
    
    liste_of_preds = liste_of_preds_classif + liste_of_preds_llm_embeddings + liste_of_preds_llm_finetuned + score
    best_model = 2
    liste_of_preds[0], liste_of_preds[best_model] = liste_of_preds[best_model], liste_of_preds[0]
    # print(liste_of_preds_llm_embeddings)
    # print(liste_of_preds[0])
    final_model = aggregate_voter(liste_of_preds,df_test)
    evaluation([final_model],df_test)
    