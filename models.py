# models.py

from sentiment_data import *
from utils import *

from collections import Counter
from numpy import exp
from random import shuffle

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.UnigramIndexer = indexer

    def get_indexer(self):
        return self.UnigramIndexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False): 
        features = Counter(sentence)
        if add_to_indexer:
            for str in sentence:
                index = self.UnigramIndexer.add_and_get_index(str)
        return features


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.BigramIndexer = indexer

    def get_indexer(self):
        return self.BigramIndexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False):
        # sliding window to generate bigrams
        sentenceLength = len(sentence)-1
        listOfBygrams = []
        for i in range(sentenceLength):
            # print(sentence[i] + ", " + sentence[i+1])
            bigram = sentence[i] + " " + sentence[i+1]
            #print(bigram)
            listOfBygrams.append(bigram)
            if add_to_indexer:
                self.BigramIndexer.add_and_get_index(bigram)
        features = Counter(listOfBygrams)
        return features

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: List[int], featurizer: FeatureExtractor):
        self.weights = weights
        self.featurizer = featurizer
        self.ypred = 0

    def predict(self, sentence: List[str]):
        # get feature, which is the vector of word quantities, from the featurizer
        # features = Counter() object
        features = self.featurizer.extract_features(sentence, True)

        # for each feature, calculate sentiment val and add to ypred
        indexer = self.featurizer.get_indexer()
        for feature, frequency in features.items():
            # get index value of weight associated with each feature
            # (for this assignment, don't have to worry about unknown words)
            index = indexer.index_of(feature)
            # multiply feature with correct weight
            #  print(index)
            self.ypred += self.weights[index] 

        # linear binary classification: if ypred > 0, return 1; else, return 0
        # make sure to reset self.ypred for the next
        if self.ypred > 0:
            # positive sentiment
            self.ypred = 0
            return 1
        else:
            self.ypred = 0
            return 0
        

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: List[int], featurizer: FeatureExtractor):
        self.weights = weights
        self.featurizer = featurizer
        # variable to hold probability, so it can be referenced during training
        self.Pofygivenx = 0

    def predict(self, sentence: List[str]):
        # zero out Pofygivenx for new prediction
        self.Pofygivenx = 0

        # get feature, which is the vector of word quantities, from the featurizer
        # features = Counter() object
        features = self.featurizer.extract_features(sentence, True)

        indexer = self.featurizer.get_indexer()
        wfx = 0
        # for each feature, calculate sentiment val and add to ypred
        for feature, frequency in features.items():
            # get index value of weight associated with each feature
            # (for this assignment, don't have to worry about unknown words)
            index = indexer.index_of(feature)
            # multiply feature with correct weight
            #  print(index)
            if index < 0 or index > len(self.weights)-1:
                print(index)
            wfx += self.weights[index] 

        # logistic regression: calculate P(y=+1|x) = e^(wfx)/(1 + e^(wfx)
        # make sure to reset self.ypred for the next
        if wfx > 709:
            print(wfx)
        self.Pofygivenx = exp(wfx) / (1 + exp(wfx))
        if wfx > 0.5:
            # positive sentiment
            return 1
        else:
            return 0
        

def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    # set up starting vals/objects
    # best ones: alpha = .03, epochs = 500 (74.43%)
    weights = [0]*20001
    alpha = .03
    alphaorig = alpha
    epochs = 500
    model = PerceptronClassifier(weights, feat_extractor)

    # perceptron training algorithm
    for t in range(epochs):
        if t != 0:
            alpha -= alphaorig / epochs
        for d in train_exs:
            # indexer is being updated with every call
            ypred = model.predict(d.words)
            yactual = d.label
            # update weights based on ypred
            if ypred == yactual:
                # no updates to weights
                pass
            elif yactual > 0:
                # increase weight vec by small amount
                # only change weights in correct indices
                indexer = model.featurizer.get_indexer()
                features = model.featurizer.extract_features(d.words, False)
                for feature, frequency in features.items():
                    # find index val associated with that specific word
                    index = indexer.index_of(feature)
                    # update weight based on perceptron algorithm
                    if index < 0:
                        print(index)
                    model.weights[index] += alpha
            else:
                # same as prev branch, but weight is updated differently
                indexer = model.featurizer.get_indexer()
                features = model.featurizer.extract_features(d.words, False)
                for feature, frequency in features.items():
                    # find index val associated with that specific word
                    index = indexer.index_of(feature)
                    if index < 0:
                        print(index)
                    # update weight based on perceptron algorithm
                    model.weights[index] -= alpha
    return model


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # set up starting vals/objects
    # best ones: alpha = .6, epochs = 200 (77.29%)
    weights = [0]*200001
    alpha = .53
    alphaorig = alpha
    epochs = 200
    model = LogisticRegressionClassifier(weights, feat_extractor)

    # perceptron training algorithm
    for t in range(epochs):
        alpha -= alphaorig / epochs
        # shuffle the data once per epoch
        shuffle(train_exs)
        for d in train_exs:
            # indexer is being updated with every call
            ypred = model.predict(d.words)
            yactual = d.label
            # update weights based on ypred
            if ypred == yactual:
                # no updates to weights
                pass
            elif yactual > 0:
                # increase weight vec by small amount
                # only change weights in correct indices
                indexer = model.featurizer.get_indexer()
                features = model.featurizer.extract_features(d.words, False)
                for feature, frequency in features.items():
                    # find index val associated with that specific word
                    index = indexer.index_of(feature)
                    # update weight based on perceptron algorithm
                    model.weights[index] += alpha * (1-model.Pofygivenx)
            else:
                # same as prev branch, but weight is updated differently
                indexer = model.featurizer.get_indexer()
                features = model.featurizer.extract_features(d.words, False)
                for feature, frequency in features.items():
                    # find index val associated with that specific word
                    index = indexer.index_of(feature)
                    # update weight based on perceptron algorithm
                    model.weights[index] -= alpha * model.Pofygivenx
    return model


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model