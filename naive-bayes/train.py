import pickle
from config import file_names
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from urllib.parse import urlparse
import warnings
import sys

import mlflow
import mlflow.sklearn


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # importing data
    df_us_train = pickle.load(open(file_names["df_us_train"], "rb"))
    df_us_trial = pickle.load(open(file_names["df_us_trial"], "rb"))
    df_us_test = pickle.load(open(file_names["df_us_test"], "rb"))

    # pre-processing
    tt = TweetTokenizer()
    df_us_train['tokenized_text'] = df_us_train['text'].str.lower().apply(lambda x: " ".join(tt.tokenize(x)))
    # df_us_train['tokenized_text'] = df_us_trial['text'].str.lower().apply(lambda x: " ".join(tt.tokenize(x)))
    df_us_test['tokenized_text'] = df_us_test['text'].str.lower().apply(lambda x: " ".join(tt.tokenize(x)))

    # fitting NB
    with mlflow.start_run():
        # based on https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html
        
        # importing parameters in case they are passed as argument
        min_count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
        print("Naive-Bayes model being trained with the following parameters")
        print('\tmin cont: %s' % min_count)
        print('\talpha: %s' % alpha)
        
        # vectorisation
        vectorizer = CountVectorizer(min_df=min_count)
        X_train_bow = vectorizer.fit_transform(df_us_train["tokenized_text"])
        X_test_bow = vectorizer.transform(df_us_test["tokenized_text"])

        # fitting classifier
        clf = MultinomialNB(alpha=alpha)
        clf.fit(X_train_bow, df_us_train["label"])

        # test metrics
        # clf.score(X_train_bow, df_us_train["label"])
        y_pred = clf.predict(X_test_bow)
        report = classification_report(df_us_test["label"], y_pred,output_dict=True)
        precision, recall, f1_score, accuracy = report['macro avg']['precision'], report['macro avg']['recall'], report['macro avg']['f1-score'], report['accuracy']
        # 'accuracy','macro f1','weighted f1','macro precision','weighted precision','macro recall','weighted recall'
        print("Results on test set:")
        print('\tprecision: %s' % precision)
        print('\trecall: %s' % recall)
        print('\tf1-score: %s' % f1_score)
        print('\taccuracy: %s' % accuracy)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("min_count", min_count)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1-score", f1_score)
        mlflow.log_metric("accuracy", accuracy)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(clf, "model", registered_model_name="Naive-Bayes_en")
        else:
            mlflow.sklearn.log_model(clf, "model")
