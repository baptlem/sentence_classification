from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models.phrases import Phrases, Phraser, ENGLISH_CONNECTOR_WORDS
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDClassifier
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")



def tf_idf_classifier(df_train,df_test,classifier=SGDClassifier(),stop_words="english"):
    pipeline = Pipeline([
    ('preprocess', FunctionTransformer(_preprocess_text)),
    ('vectorizer',  TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5, stop_words=stop_words)),
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



def llm_embedding(model_name,df_train,df_test,k=25):
    # TODO replace by sklearn knn
    model = SentenceTransformer(model_name)
    embeddings_test = model.encode(list(df_test["sentences"]))
    embedding_train = model.encode(list(df_train['sentences']))
    results = cosine_similarity(embedding_train,embeddings_test).transpose()
    highest_indices = np.argsort(results)[:,-k:]
    result = []
    for i in range(len(highest_indices)):
        highest_indices[i] = df_train.loc[highest_indices[i],'labels']
        result.append(_most_common_element(highest_indices[i]))
    return result
        
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
    

def finetuned_llm():
    pass



