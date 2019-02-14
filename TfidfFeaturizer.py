from Featurizer import Featurizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# This is a subclass that extends the abstract class Featurizer.
class TfidfFeaturizer(Featurizer):
    # The abstract method from the base class is implemeted here to return tf-idf features
    def getFeatureRepresentation(self, X_train, X_val):
        # tune params??
        count_vect = TfidfVectorizer(lowercase=True, stop_words='english', max_features=3000)
        # fit_transform is equivalent to fit followed by transform, but more efficiently implemented.
        X_train_counts = count_vect.fit_transform(X_train)
        X_val_counts = count_vect.transform(X_val)
        return X_train_counts, X_val_counts
