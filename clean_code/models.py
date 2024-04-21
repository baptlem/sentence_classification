from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import SGDClassifier
import warnings
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.utils import shuffle
warnings.filterwarnings("ignore")


"""Not all the functions here where used in the final submission, but they show some of the tests we made"""


def compare_to_class(model, df_train, df_test, embedding_name):
    """
    Train a classifier on the cosine similarity between the embeddings of the themes and the embeddings of the sentences.

    Args:
        model (sklearn.base.BaseEstimator): The classifier model.
        df_train (pd.DataFrame): The training dataframe containing the sentences.
        df_test (pd.DataFrame): The test dataframe containing the sentences.
        embedding_name (str): The name of the embedding model to be used.

    Returns:
        numpy.ndarray: The predicted labels for the test data.
    """
    embedding_model = SentenceTransformer(embedding_name)
    theme = ["Politics", "Health", "Finance", "Travel", "Food", "Education", "Environment", "Fashion", "Science", "Sports", "Technology", "Entertainment"]
    embeddings_theme = embedding_model.encode(theme)
    embeddings_train = embedding_model.encode(list(df_train['sentences']))
    embedding_test = embedding_model.encode(list(df_test['sentences']))
    results = cosine_similarity(embeddings_theme, embeddings_train).transpose()
    results_test = cosine_similarity(embeddings_theme, embedding_test).transpose()
    model.fit(results, df_train['labels'])
    predict = model.predict(results_test)
    return predict


def tf_idf_classifier(df_train, df_test, classifier=SGDClassifier(), stop_words="english", train_vocab=None):
    """
    Trains a TF-IDF classifier using the provided training data and predicts labels for the test data.

    Args:
        df_train (pandas.DataFrame): The training data DataFrame containing 'sentences' and 'labels' columns.
        df_test (pandas.DataFrame): The test data DataFrame containing 'sentences' column.
        classifier (object, optional): The classifier model to use. Defaults to SGDClassifier().
        stop_words (str or list, optional): The stop words to be removed during text preprocessing. Defaults to "english".
        train_vocab (list, optional): The vocabulary to use for vectorization. Defaults to None.

    Returns:
        numpy.ndarray: The predicted labels for the test data.

    """
    vocab = None
    if train_vocab is not None:
        vocab = _create_vocab(train_vocab, stop_words="english")
    pipeline = Pipeline([
        ('preprocess', FunctionTransformer(_preprocess_text)),
        ('vectorizer',  TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words=stop_words, vocabulary=vocab, strip_accents='unicode', decode_error='replace')),
        ('selector',  SelectKBest(f_classif, k=500)),
        ("classifier", classifier)
    ])
    X_train = df_train.loc[:, "sentences"]
    y_train = df_train.loc[:, "labels"]
    pipeline.fit(X_train, y_train)
    predicted_label = pipeline.predict(df_test.loc[:, "sentences"])
    return predicted_label 
    
def _preprocess_text(X):
    """function to preprocess text data, tokenizes and lemmatizes the text data"""
    
    X = X.str.lower().apply(word_tokenize)
    lemmatizer = WordNetLemmatizer()
    X = X.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    return X.apply(lambda x:' '.join(x))

def _create_vocab(df_train,stop_words="english"):
    """function to create a vocabulary from the training data based on the similarity
    between each element of the vocabulary and the themes"""
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    theme = ["Politics","Health","Finance","Travel","Food","Education","Environment","Fashion","Science","Sports","Technology","Entertainment"]
    threshold =  [0.405,0.38,0.37,0.37,0.4,0.41,0.4,0.40,0.43,0.395,0.408,0.42]
    
    pipeline = Pipeline([
    ('preprocess', FunctionTransformer(_preprocess_text)),
    ('vectorizer',  TfidfVectorizer(sublinear_tf=True, stop_words=stop_words))
    ])
    
    predicted_label = pipeline.fit_transform(df_train.loc[:,"sentences"])
    vectorizer = pipeline.named_steps['vectorizer']
    vocabulary = vectorizer.vocabulary_
    embedding_voc= model.encode(list(vocabulary.keys()))
    embedding_theme = model.encode(theme)
    results = cosine_similarity(embedding_voc,embedding_theme)
    vectors = np.array(results) 
    indices = np.where(np.any(vectors > threshold, axis=1))[0]
    return list(np.array(list(vocabulary.keys()))[indices])


def finetuned_llm(model_name_llm, model_name_classifier, df_train, df_test):
    """
    Fine-tunes a language model and a classifier using the provided training data and predicts labels for the test data.

    Args:
        model_name_llm (str): The name of the language model to be used for encoding sentences.
        model_name_classifier: The classifier model to be used for training and prediction.
        df_train (pandas.DataFrame): The training data containing sentences and corresponding labels.
        df_test (pandas.DataFrame): The test data containing sentences for prediction.

    Returns:
        numpy.ndarray: The predicted labels for the test data. 
    """
    df_train = shuffle(df_train)
    model_llm = SentenceTransformer(model_name_llm)
    embeddings_test = model_llm.encode(list(df_test["sentences"]))
    embedding_train = model_llm.encode(list(df_train['sentences']))
    model_name_classifier.fit(embedding_train, df_train['labels'])
    predicted_label = model_name_classifier.predict(embeddings_test)
    return predicted_label
    



