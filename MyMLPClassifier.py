from Classifier import Classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier


# This is a subclass that extends the abstract class Classifier.
class MyMLPClassifier(Classifier):
    # The abstract method from the base class is implemeted here to return multinomial naive bayes classifier
    def buildClassifier(self, X_features, Y_train):
        # clf = MLPClassifier()
        # clf.fit(X_features, Y_train)
        clf = MLPClassifier(hidden_layer_sizes=(80,), max_iter=500).fit(X_features, Y_train)
        return clf
