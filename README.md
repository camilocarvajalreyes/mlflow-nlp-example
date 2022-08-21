# MLflow example with NLP models

**NLP example of model management using MLflow.**

We will use the Multilingual Emoji Prediction dataset (Barbieri et al. 2010), which consists of tweets in English and Spanish that originally had a single emoji, which is later used as a tag. Test and trial sets can be downloaded [here](https://github.com/fvancesco/Semeval2018-Task2-Emoji-Detection/blob/master/dataset/Semeval2018-Task2-EmojiPrediction.zip?raw=true), but the train set needs to be downloaded using a [twitter crawler](https://github.com/fra82/twitter-crawler/blob/master/semeval2018task2TwitterCrawlerHOWTO.md). The goal is to predict that single emoji that was originally in the tweet using the text in it (out of a fixed set of possible emojis, 20 for English and 19 for Spanish).

The models used in this repository were implemented for the course [CC5205 Data Mining](https://github.com/dccuchile/CC5205). A deeper analysis can be found in [github.com/furrutiav/data-mining-2022](https://github.com/furrutiav/data-mining-2022). Credit to the colaborators of that project. This work is a toy version of a larger project between UFRO and ESO. This collaboration attempts to deploy ML models for error troubleshooting at Paranal Observatory trough software logs, while folloeing MLOps principles.

## Instructions

**Install requirements for this example**

```pip install -r requirements```

Note that when deploying a model with conda, another conda env will be created based on library versions as detected by _mlflow_.


**MLflow user interface for model comparison**

Run

```mlflow ui```

and go to

http://localhost:5000/

**Running and testing model with different parameters**

```conda run -n env_name python naive-bayes/train.py <min_count> <alpha>```

Or, if using Anaconda prompt on Windows:

```python naive-bayes/train.py <min_count> <alpha>```

It will print metrics on the test set and will save the runs. Runs are available at `mlflow ui`. <min_count> and <alpha> should be replaced with the desired values (a positive integer and a float between 0 and 1 respectively), without '<>'.

**Serving model**

```mlflow models serve -m mlruns/0/b093c14327d7411a829b18102c6b8f8e/artifacts/Naive-Bayes_model -p 1234```
