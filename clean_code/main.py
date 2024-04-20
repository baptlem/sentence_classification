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
    
    df_train,df_test = import_data("data/test.txt","data/annotated_test.txt")#test_shuffle.txt annotated_test.txt
    

    weights_ridge =  {6:(95/140),1:(95/99),5:(95/105),0:(95/111),7: (95/97),9: (95/98),10:(95/86),2: (95/81),11:(95/90),3:(95/81),8:(95/59),4:(95/93)}
    weights_sgd = {6:(95/137),1:(95/124),5:(95/63),0:(95/111),7: (95/96),9: (95/99),10:(95/93),2: (95/94),11:(95/95),3:(95/63),8:(95/58),4:(95/77)}
    weights_svc = {6:(95/136),1:(95/101),5:(95/101),0:(95/115),7: (95/96),9: (95/99),10:(95/92),2: (95/89),11:(95/89),3:(95/68),8:(95/61),4:(95/93)}
    weights_extratree = {6:(95/132),1:(95/116),5:(95/98),0:(95/124),7: (95/94),9: (95/95),10:(95/102),2: (95/84),11:(95/94),3:(95/63),8:(95/54),4:(95/84)}

    liste_of_preds_llm_finetuned = []

    model_name_classifier = [
                    SGDClassifier(loss='log_loss', penalty='l2',alpha=0.005, random_state=42,class_weight=weights_sgd
                                )
                    ,RidgeClassifier(alpha=10.0, solver="sparse_cg",class_weight=weights_ridge
                                     )
                    ,LinearSVC(C=0.1, dual=False,class_weight=weights_svc
                                )
                    ,ExtraTreesClassifier(n_estimators=500, random_state=42,class_weight=weights_extratree
                                        )
                    ] 
    


    for model_name_classifier in tqdm(model_name_classifier):
        # pass
        predicted,predicted_proba = finetuned_llm('sentence-transformers/all-MiniLM-L6-v2',model_name_classifier,df_train,df_test)
        liste_of_preds_llm_finetuned.append(predicted)
    # evaluation(liste_of_preds_llm_finetuned,df_test)
    
    liste_of_preds =  liste_of_preds_llm_finetuned
    print(len(liste_of_preds))
    best_model = 1
    liste_of_preds[0], liste_of_preds[best_model] = liste_of_preds[best_model], liste_of_preds[0]

    final_model = aggregate_voter(liste_of_preds)
    # save_prediction(final_model,"lol6")
    
    evaluation([final_model],df_test)
    print(pd.Series(final_model).value_counts())
    
    # filename = "results/only_last.csv"
    # dico = pd.read_json("data/train.json")
    # corresp_labels = {str(i):col for i,col in enumerate(dico.columns)}
    # with open(filename, "w", newline="") as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',')  
    #     writer.writerow(["ID", "Label"])  
    #     for i, item in enumerate(final_model, start=0):
    #         writer.writerow([i, corresp_labels[str(item)]]) 
            