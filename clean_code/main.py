from utilities import _most_common_element,evaluation,import_data,aggregate_voter,save_prediction,load_prediction,aggregate_voter_proba
from models import tf_idf_classifier,llm_embedding,finetuned_llm,bert_score,llm_embedding_centroid,compare_to_class
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
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier,RidgeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import LinearSVC
from tqdm import tqdm
import warnings
import csv
from xgboost import XGBClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    
    #backtrans_and_free.txt: 0.61/0.46/0.8/0.79 (biaisé sur 11  pour les deux premiers)
    #dataset_de_mort.txt: 0.61/0.46/0.8/0.72 (biaisé sur 11  pour les deux premiers mais plus)
    #llama2_train_set_0804.txt: 0.49/0.42/0.8/0.77 (biaisé sur 9  pour le deuxième et le 5 je crois pour le premier)
    #0804_llama.txt: 0.56/0.38/0.81/0.81 (biaisé sur 11  pour les deux premiers)
    # TODO data quality, feature hashing,other embeddings,function to save predictions,lstm?
    
    df_train,df_test = import_data("data/dataset_de_mort_and_food_custom.txt","data/annotated_test.txt")#test_shuffle.txt annotated_test.txt
    
    liste_of_classifier = [
                        # SGDClassifier(loss='log_loss', penalty='l2',alpha=0.005, random_state=42)
                        RidgeClassifier(alpha=10.0, solver="sparse_cg")
                        #  ,LinearSVC(C=0.1, dual=False)
                        # ,ExtraTreesClassifier(n_estimators=500, random_state=42)
                        # ,ComplementNB(alpha=1.0)
                        # , XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',n_estimators=20,min_child_weight=2),

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
        # predicted = tf_idf_classifier(df_train,df_test,classifier=model,stop_words=None,train_vocab=None)#
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
    
    pred_direct = []
    weights_ridge1 =  {6:(95/118),1:(95/95),5:(95/110),0:(95/109),7: (95/100),9: (95/122),10:(95/69),2: (95/95),11:(95/66),3:(95/83),8:(95/78),4:(95/95)}
    weights_svc1 =  {6:(95/117),1:(95/111),5:(95/110),0:(95/108),7: (95/99),9: (95/123),10:(95/78),2: (95/84),11:(95/75),3:(95/76),8:(95/75),4:(95/84)}
    weights_extratree1 =  {6:(95/96),1:(95/108),5:(95/115),0:(95/105),7: (95/95),9: (95/108),10:(95/105),2: (95/85),11:(95/86),3:(95/60),8:(95/94),4:(95/83)}

    liste_of_classifier2 = [
                        # SGDClassifier(loss='log_loss', penalty='l2',alpha=0.005, random_state=42)
                        RidgeClassifier(alpha=10.0, solver="sparse_cg")
                        ,LinearSVC(C=0.1, dual=False,class_weight = weights_svc1)
                        ,ExtraTreesClassifier(n_estimators=500, random_state=42,class_weight=weights_extratree1)
                        # ,ComplementNB(alpha=1.0)
                        # , XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',n_estimators=20,min_child_weight=2),

                        #    ,LogisticRegression(C=5, max_iter=1000)
                        #    ,RidgeClassifier(alpha=1.0, solver="sparse_cg")
                        ]
    for model in tqdm(liste_of_classifier2):
        # pass
        pred_direct.append(compare_to_class(model,df_train,df_test,'sentence-transformers/all-MiniLM-L6-v2'))
    evaluation(pred_direct,df_test)
    
    liste_of_preds_llm_embeddings_centroid = []
    for model in tqdm(liste_of_llm_embeddings):
        pass
        # predicted = llm_embedding_centroid(model,df_train,df_test)
        # liste_of_preds_llm_embeddings_centroid.append(predicted)
    # save_prediction(liste_of_preds_llm_embeddings_centroid,"llm_embeddings_centroid_e5")
    # liste_of_preds_llm_embeddings_centroid = load_prediction("llm_embeddings_centroid_e5")
    # evaluation(liste_of_preds_llm_embeddings_centroid,df_test)
    
    weights = None
    probability = False
    weights_ridge =  {6:(95/148),1:(95/106),5:(95/109),0:(95/112),7: (95/96),9: (95/96),10:(95/87),2: (95/86),11:(95/89),3:(95/76),8:(95/58),4:(95/77)}
    weights_sgd = {6:(95/144),1:(95/103),5:(95/112),0:(95/116),7: (95/96),9: (95/104),10:(95/92),2: (95/90),11:(95/89),3:(95/55),8:(95/61),4:(95/78)}
    weights_svc = {6:(95/146),1:(95/123),5:(95/111),0:(95/115),7: (95/97),9: (95/97),10:(95/89),2: (95/90),11:(95/86),3:(95/66),8:(95/62),4:(95/58)}
    weights_extratree = {6:(95/137),1:(95/116),5:(95/106),0:(95/133),7: (95/95),9: (95/93),10:(95/100),2: (95/83),11:(95/97),3:(95/60),8:(95/52),4:(95/68)}

    liste_of_preds_llm_finetuned = []
    model_name_classifier = [
                        # XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',n_estimators=20,min_child_weight=2),
                        # RandomForestClassifier(n_estimators=1000, random_state=42,class_weight=weights),
                        SGDClassifier(loss='log_loss', penalty='l2',alpha=0.005, random_state=42,class_weight=weights_ridge)
                        ,RidgeClassifier(alpha=10.0, solver="sparse_cg",class_weight=weights_sgd)
                         ,LinearSVC(C=0.1, dual=False,class_weight=weights_svc)
                         ,ExtraTreesClassifier(n_estimators=500, random_state=42,class_weight=weights_extratree)
                        #  ,AdaBoostClassifier(n_estimators=1000, random_state=0)
                        # ,GaussianProcessClassifier(kernel=1.0 * RBF(1.0),random_state=0)
                        # ,ExtraTreesClassifier(n_estimators=100, random_state=0)
                        # ,SVC(C=0.1,class_weight=weights)
                        #   ,LogisticRegression(C=0.5, max_iter=1000,class_weight=weights)
                        # ,ComplementNB(alpha=1.0)
                        ] 
    # TODO rajouter d'autres modèles plus efficaces
    #"mixedbread-ai/mxbai-embed-large-v1" 26 minutes outch 
    liste_of_preds_proba_llm_finetuned = []                             
    for model_name_classifier in tqdm(model_name_classifier):
        # pass
        predicted,predicted_proba = finetuned_llm('sentence-transformers/all-MiniLM-L6-v2',model_name_classifier,df_train,df_test)
        liste_of_preds_llm_finetuned.append(predicted)
        liste_of_preds_proba_llm_finetuned.append(predicted_proba)
    # save_prediction(liste_of_preds_llm_finetuned,"mixedbread_xgboost")
    # liste_of_preds_llm_finetuned = load_prediction("mixedbread_xgboost")
    evaluation(liste_of_preds_llm_finetuned,df_test)
    
    #liste_of_preds_classif + liste_of_preds_llm_embeddings + liste_of_preds_llm_finetuned + score +
    liste_of_preds = liste_of_preds_classif + liste_of_preds_llm_embeddings + liste_of_preds_llm_finetuned + score + liste_of_preds_llm_embeddings_centroid + pred_direct
    print(len(liste_of_preds))
    best_model = 0
    liste_of_preds[0], liste_of_preds[best_model] = liste_of_preds[best_model], liste_of_preds[0]
    # print(liste_of_preds_llm_embeddings)
    # print(liste_of_preds[0])
    final_model = aggregate_voter(liste_of_preds)
    save_prediction(final_model,"lol3")
    # final_model_proba = aggregate_voter_proba(liste_of_preds_proba_llm_finetuned)
    evaluation([final_model],df_test)
    # evaluation([final_model_proba],df_test)
    print(pd.Series(final_model).value_counts())
    
    # filename = "results/10.csv"
    # dico = pd.read_json("data/train.json")
    # corresp_labels = {str(i):col for i,col in enumerate(dico.columns)}
    # with open(filename, "w", newline="") as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',')  
    #     writer.writerow(["ID", "Label"])  
    #     for i, item in enumerate(final_model, start=0):
    #         writer.writerow([i, corresp_labels[str(item)]]) 
            