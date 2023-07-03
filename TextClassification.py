import os
import string

# Data Handling and Processing
import pandas as pd
import numpy as np
import re
from scipy import interp

# NLP Packages
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import nltk

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

from joblib import dump, load


# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier

# Scikit Learn packages
from sklearn.base import clone
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import (
    KFold,
    cross_validate,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

filename = "articles.csv"


def read_csv():
    return pd.read_csv(filename)


def preprocess(df):
    # check for missing value
    df.isna().sum()

    # Remove special characters
    df["article_temp"] = df["article"].replace("\n", " ")
    df["article_temp"] = df["article_temp"].replace("\r", " ")

    # Remove punctuation signs and lowercase all
    df["article_temp"] = df["article_temp"].str.lower()
    df["article_temp"] = df["article_temp"].str.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    def fwpt(each):
        tag = pos_tag([each])[0][1][0].upper()
        hash_tag = {
            "N": wordnet.NOUN,
            "R": wordnet.ADV,
            "V": wordnet.VERB,
            "J": wordnet.ADJ,
        }
        return hash_tag.get(tag, wordnet.NOUN)

    def lematize(text):
        tokens = nltk.word_tokenize(text)
        ax = ""
        for each in tokens:
            if each not in stop_words:
                ax += lemmatizer.lemmatize(each, fwpt(each)) + " "
        return ax

    df["article_temp"] = df["article_temp"].apply(lematize)

    return df;


def get_classification(input_text):

    df = read_csv();
    df = preprocess(df);

    X_train, X_test, y_train, y_test = train_test_split(
        df["article_temp"], df["category"], test_size=0.2, random_state=9
    )

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


    # vecotrize
    vector = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=3, max_df=1.0, max_features=10000
    )


    # fit modal
    def fit_model(model, model_name):
        line = Pipeline([("vectorize", vector), (model_name, model)])

        output = cross_validate(
            line,
            X_train,
            y_train,
            cv=KFold(shuffle=True, n_splits=3, random_state=9),
            scoring=("accuracy", "f1_weighted", "precision_weighted", "recall_weighted"),
            return_train_score=True,
        )
        return output


    dectree = fit_model(DecisionTreeClassifier(), "DTree")
    ridge = fit_model(RidgeClassifier(), "Ridge")
    bayes = fit_model(MultinomialNB(), "NB")

    dt = pd.DataFrame.from_dict(dectree)
    rc = pd.DataFrame.from_dict(ridge)
    bc = pd.DataFrame.from_dict(bayes)


    l1 = [bc, rc, dt]
    l2 = ["NB", "Ridge", "DT"]

    for each, tag in zip(l1, l2):
        each["model"] = [tag, tag, tag]

    joined_output = pd.concat([bc, rc, dt])

    print(dectree)

    print(ridge)

    print(bayes)

    relevant_measures = list(
        [
            "test_accuracy",
            "test_precision_weighted",
            "test_recall_weighted",
            "test_f1_weighted",
        ]
    )

    dec_tree_metrics = joined_output.loc[joined_output.model == "DT"][relevant_measures]
    nb_metrics = joined_output.loc[joined_output.model == "NB"][relevant_measures]
    r_metrics = joined_output.loc[joined_output.model == "Ridge"][relevant_measures]

    print(dec_tree_metrics)

    print(nb_metrics)

    print(r_metrics)

    metrics_ = [dec_tree_metrics, nb_metrics, r_metrics]
    names_ = ["Decision Tree", "Naive Bayes", "Ridge Classifier"]

    for scores, namess in zip(metrics_, names_):
        print(f"{namess} Mean Metrics:")
        print(scores.mean())
        print("  ")

    # selection of modal

    # Join training and test datasets
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])


    def create_and_fit(clf, x, y):
        best_clf = clf
        pipeline = Pipeline([("vectorize", vector), ("model", best_clf)])
        return pipeline.fit(x, y)


    # Create model
    CLASSYfier = create_and_fit(MultinomialNB(), X, y)

    print(CLASSYfier.classes_)
    CLASSYfier.predict_proba([input_text]);
    print(CLASSYfier.predict([input_text]))
    return CLASSYfier.predict([input_text])[0];


get_classification("Raising equity capital via Private Investments in Public Equity (PIPEs) has been rising in popularity, matching Seasoned Equity Offerings (SEOs). We use over 10,000 PIPEs in a global setting during 1995–2015 to assess how and through which channels institutional frameworks affect the issuers' performance. We document a significant decline in the market reaction, especially during 2004–2015 and find that firms issuing equity via PIPEs have significantly worse fundamentals. We also show that country governance matters as issuing firms operating in countries with better regulatory environments outperform others. Finally, we find that regulatory enforcement is a plausible underlying channel for the positive effect of the institutional frameworks on PIPEs performance.");