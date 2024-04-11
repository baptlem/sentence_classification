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
from sklearn.linear_model import LogisticRegression, SGDClassifier,RidgeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC
from tqdm import tqdm
import warnings
import csv
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    
    #backtrans_and_free.txt: 0.61/0.46/0.8/0.79 (biaisé sur 11  pour les deux premiers)
    #dataset_de_mort.txt: 0.61/0.46/0.8/0.72 (biaisé sur 11  pour les deux premiers mais plus)
    #llama2_train_set_0804.txt: 0.49/0.42/0.8/0.77 (biaisé sur 9  pour le deuxième et le 5 je crois pour le premier)
    #0804_llama.txt: 0.56/0.38/0.81/0.81 (biaisé sur 11  pour les deux premiers)
    # TODO data quality, feature hashing,other embeddings,function to save predictions,lstm?
    
    df_train,df_test = import_data("data/dataset_de_mort_and_food_custom.txt","data/annotated_test.txt")#test_shuffle.txt annotated_test.txt
    
    sgd_classifier = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42)
    liste_of_classifier = [sgd_classifier,ComplementNB(alpha=0.1)
                           ,LinearSVC(C=0.1, dual=False)
                        #    ,LogisticRegression(C=5, max_iter=1000)
                        #    ,RidgeClassifier(alpha=1.0, solver="sparse_cg")
                           ]
    liste_of_llm_embeddings = ['sentence-transformers/all-MiniLM-L6-v2']#"intfloat/multilingual-e5-base"['sentence-transformers/all-MiniLM-L6-v2']#["Salesforce/SFR-Embedding-Mistral",'sentence-transformers/all-MiniLM-L6-v2',"mixedbread-ai/mxbai-embed-large-v1"]
    list_of_finetuned_llm = ['sentence-transformers/all-MiniLM-L6-v2']
    
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
        pass
        # predicted = llm_embedding_centroid(model,df_train,df_test)
        # liste_of_preds_llm_embeddings_centroid.append(predicted)
    # save_prediction(liste_of_preds_llm_embeddings_centroid,"llm_embeddings_centroid_e5")
    # liste_of_preds_llm_embeddings_centroid = load_prediction("llm_embeddings_centroid_e5")
    # evaluation(liste_of_preds_llm_embeddings_centroid,df_test)
    
    weights = None
    # weights =  {6:95/149,1:95/119,5:95/111,0:95/118,7: 95/97,9: 95/97,10:95/90,2: 95/86,11:95/85,3: 95/72,8: 95/57,4: 95/59}
    liste_of_preds_llm_finetuned = []
    model_name_classifier = [
                        # XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',n_estimators=20,min_child_weight=2),
                        # RandomForestClassifier(n_estimators=100, random_state=42),
                        SGDClassifier(loss='hinge', penalty='l2',alpha=0.005, random_state=42,class_weight=weights)
                        ,RidgeClassifier(alpha=10.0, solver="sparse_cg",class_weight=weights)
                         ,LinearSVC(C=0.1, dual=False,class_weight=weights)
                        ] 
    #"mixedbread-ai/mxbai-embed-large-v1" 26 minutes outch                               
    for model_name_classifier in tqdm(model_name_classifier):
        # pass
        predicted = finetuned_llm('sentence-transformers/all-MiniLM-L6-v2',model_name_classifier,df_train,df_test)
        liste_of_preds_llm_finetuned.append(predicted)
    # save_prediction(liste_of_preds_llm_finetuned,"mixedbread_xgboost")
    # liste_of_preds_llm_finetuned = load_prediction("mixedbread_xgboost")
    evaluation(liste_of_preds_llm_finetuned,df_test)
    
    #liste_of_preds_classif + liste_of_preds_llm_embeddings + liste_of_preds_llm_finetuned + score +
    liste_of_preds = liste_of_preds_classif + liste_of_preds_llm_embeddings + liste_of_preds_llm_finetuned + score + liste_of_preds_llm_embeddings_centroid
    print(len(liste_of_preds))
    best_model = 0
    liste_of_preds[0], liste_of_preds[best_model] = liste_of_preds[best_model], liste_of_preds[0]
    # print(liste_of_preds_llm_embeddings)
    # print(liste_of_preds[0])
    final_model = aggregate_voter(liste_of_preds)
    evaluation([final_model],df_test)
    print(pd.Series(final_model).value_counts())
    
    # filename = "result3.csv"
    # dico = pd.read_json("data/train.json")
    # corresp_labels = {str(i):col for i,col in enumerate(dico.columns)}
    # with open(filename, "w", newline="") as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',')  
    #     writer.writerow(["ID", "Label"])  
    #     for i, item in enumerate(final_model, start=0):
    #         writer.writerow([i, corresp_labels[str(item)]]) 
            