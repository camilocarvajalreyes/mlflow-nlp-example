"""Defining a custom flavor for wrapping both tokenization and inference within a mlflow model class"""
import mlflow

class NaiveBayesModelWrapper(mlflow.pyfunc.PythonModel):
    """Class that will allows us to use save it for model management with mlflow"""
    def __init__(self, classifier, tokenizer, vectorizer):
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer

    def predict(self, context, model_input):
        input_tokenized = self.tokenizer(model_input)
        input_vectorized = self.vectorizer.transform(input_tokenized)
        return self.classifier.predict(input_vectorized)
