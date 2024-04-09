
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
import warnings
import pickle
warnings.filterwarnings("ignore")

DIRECTORY = "clean_code/saved_predictions/"

def save_prediction(pred,name):
    with open(DIRECTORY + name, 'wb') as f:
        pickle.dump(pred, f)
     
def load_prediction(name):
    file_path = DIRECTORY + name
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def evaluation(predicted_label,df_test):
    if len(predicted_label) == 0:
        return
    if len(predicted_label) == 1:
        report = classification_report(df_test.loc[:, "labels"], pd.Series(predicted_label[0]))
        print(report)
        cm = confusion_matrix(df_test.loc[:, "labels"], pd.Series(predicted_label[0]))
        df_cm = pd.DataFrame(cm, index=list(range(12)), columns=list(range(12)))
        sn.set(font_scale=1)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, cmap='coolwarm', linewidth=0.5)
        plt.show()
        return
    fig, axs = plt.subplots(len(predicted_label), figsize=(10, 10))
    for i, pred in enumerate(predicted_label):
        report = classification_report(df_test.loc[:, "labels"], pd.Series(pred))
        print(report)
        cm = confusion_matrix(df_test.loc[:, "labels"], pd.Series(pred))
        df_cm = pd.DataFrame(cm, index=list(range(12)), columns=list(range(12)))
        sn.set(font_scale=1)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, cmap='coolwarm', linewidth=0.5, ax=axs[i])
    plt.show()


def import_data(train_path,test_path):
    df_train = pd.read_csv(train_path,names=['labels','sentences'],sep='\t')
    df_train = df_train.drop_duplicates(subset='sentences').reset_index(drop=True)
    df_test = pd.read_csv(test_path, sep=';',names=['labels','sentences'])
    # print(df_train.shape,df_test.shape)
    return df_train,df_test

def aggregate_voter(list_of_preds,df_test):
    testset = df_test.loc[:,"labels"]
    assembly_result = []
    for i in range(len(testset)):
        # print([voter[i] for voter in list_of_preds],testset[i],most_common_element([voter[i] for voter in list_of_preds]))
        assembly_result.append(_most_common_element([voter[i] for voter in list_of_preds]))   
    return assembly_result

def _most_common_element(lst):
    counts = {}
    for element in lst:
        if element in counts:
            counts[element] += 1
        else:
            counts[element] = 1
    max_count = max(counts.values())
    most_common_elements = [key for key, value in counts.items() if value == max_count]
    most_common_element = None
    for element in lst:
        if element in most_common_elements:
            most_common_element = element
            break
    
    return most_common_element
