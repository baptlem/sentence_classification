from utilities import _most_common_element,evaluation,import_data,aggregate_voter,save_prediction,load_prediction
from models import tf_idf_classifier,llm_embedding,finetuned_llm,bert_score,llm_embedding_centroid
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
import csv
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    
    # TODO data quality, feature hashing,other embeddings,function to save predictions,lstm?
    
    df_train,df_test = import_data("data/dataset_de_mort.txt","data/test_shuffle.txt")#test_shuffle.txt annoted_test.txt
    
    sgd_classifier = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42)
    liste_of_classifier = [sgd_classifier,NearestCentroid()
                           #,ComplementNB(),KNeighborsClassifier(),NearestCentroid(),LinearSVC()
                           ]
    liste_of_llm_embeddings = ['sentence-transformers/all-MiniLM-L6-v2']#"intfloat/multilingual-e5-base"['sentence-transformers/all-MiniLM-L6-v2']#["Salesforce/SFR-Embedding-Mistral",'sentence-transformers/all-MiniLM-L6-v2',"mixedbread-ai/mxbai-embed-large-v1"]
    list_of_finetuned_llm = []
    
    # score = [bert_score(df_train,df_test)]
    score = []
    
    liste_of_preds_classif = []
    for model in tqdm(liste_of_classifier):
        pass
        # predicted = tf_idf_classifier(df_train,df_test,classifier=model,train_vocab=df_train)
        # liste_of_preds_classif.append(predicted)
    # evaluation(liste_of_preds_classif,df_test)
    # save_prediction(liste_of_preds_classif,"classif")
    
    liste_of_preds_llm_embeddings = []
    for model in tqdm(liste_of_llm_embeddings):
        pass
        # predicted = llm_embedding(model,df_train,df_test)
        # liste_of_preds_llm_embeddings.append(predicted)
    # save_prediction(liste_of_preds_llm_embeddings,"llm_embeddings")
    # liste_of_preds_llm_embeddings = load_prediction("llm_embeddings")
    # evaluation(liste_of_preds_llm_embeddings,df_test)
    
    
    liste_of_preds_llm_embeddings_centroid = []
    for model in tqdm(liste_of_llm_embeddings):
        # pass
        predicted = llm_embedding_centroid(model,df_train,df_test)
        liste_of_preds_llm_embeddings_centroid.append(predicted)
    # save_prediction(liste_of_preds_llm_embeddings_centroid,"llm_embeddings_centroid_e5")
    # liste_of_preds_llm_embeddings_centroid = load_prediction("llm_embeddings_centroid_e5")
    # evaluation(liste_of_preds_llm_embeddings_centroid,df_test)
    

    liste_of_preds_llm_finetuned = []
    for model in tqdm(list_of_finetuned_llm):
        pass
        # predicted = finetuned_llm(model,df_train,df_test)
        # liste_of_preds_llm_finetuned.append(predicted)
    # evaluation(liste_of_preds_llm_finetuned,df_test)
    
    #liste_of_preds_classif + liste_of_preds_llm_embeddings + liste_of_preds_llm_finetuned + score +
    liste_of_preds = liste_of_preds_classif + liste_of_preds_llm_embeddings + liste_of_preds_llm_finetuned + score + liste_of_preds_llm_embeddings_centroid
    print(len(liste_of_preds))
    best_model = 0
    liste_of_preds[0], liste_of_preds[best_model] = liste_of_preds[best_model], liste_of_preds[0]
    # print(liste_of_preds_llm_embeddings)
    # print(liste_of_preds[0])
    final_model = aggregate_voter(liste_of_preds)
    # evaluation([final_model],df_test)
    print(pd.Series(final_model).value_counts())
    
    filename = "result1.csv"
    dico = pd.read_json("data/train.json")
    corresp_labels = {str(i):col for i,col in enumerate(dico.columns)}
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')  
        writer.writerow(["ID", "Label"])  
        for i, item in enumerate(final_model, start=0):
            writer.writerow([i, corresp_labels[str(item)]]) 