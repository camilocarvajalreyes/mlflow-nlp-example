import pickle
from config import file_names
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


if __name__ == "__main__":

    # importing data
    df_us_train = pickle.load(open(file_names["df_us_train"], "rb"))
    df_us_trial = pickle.load(open(file_names["df_us_trial"], "rb"))
    df_us_test = pickle.load(open(file_names["df_us_test"], "rb"))

    # pre-processing
    tt = TweetTokenizer()
    df_us_train['tokenized_text'] = df_us_train['text'].str.lower().apply(lambda x: " ".join(tt.tokenize(x)))
    # df_us_train['tokenized_text'] = df_us_trial['text'].str.lower().apply(lambda x: " ".join(tt.tokenize(x)))
    df_us_test['tokenized_text'] = df_us_test['text'].str.lower().apply(lambda x: " ".join(tt.tokenize(x)))

    # vectorisation
    vectorizer = CountVectorizer(min_df=5)
    X_train_bow = vectorizer.fit_transform(df_us_train["tokenized_text"])
    X_test_bow = vectorizer.transform(df_us_test["tokenized_text"])

    # fitting NB
    clf = MultinomialNB()
    clf.fit(X_train_bow, df_us_train["label"])

    # test metrics
    # clf.score(X_train_bow, df_us_train["label"])
    y_pred = clf.predict(X_test_bow)
    report = classification_report(df_us_test["label"], y_pred,output_dict=True)
    precission, recall, f1_score, accuracy = report['macro avg']['precision'], report['macro avg']['recall'], report['macro avg']['f1-score'], report['accuracy']
    # 'accuracy','macro f1','weighted f1','macro precision','weighted precision','macro recall','weighted recall'
    print(precission, recall, f1_score, accuracy)
