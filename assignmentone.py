'''
Importing all the dependences
'''
import random
import collections
import string
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus.europarl_raw import german, english, italian, french, dutch
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer


'''
Defining the variables
'''
e_languages = [english]                                 # target languages
ne_languages = [german, italian, french, dutch]         # not target languages
labeled_phrases = []                                    # list of labeled phrases
all_words = []                                          # list of all the words in the corpus
stopwords_universal = []                                # list of stopwords for every language supported
stemmer = PorterStemmer()                               # stemmer from nltk
lemmatizer = WordNetLemmatizer()                        # lemmatizer from nltk


'''
    This function is responsible for stemming and lemmatization
'''
def stemmingAndLemmatize(word):
    return lemmatizer.lemmatize(stemmer.stem(word))

'''
    This function is responsible for removing the stop words and tokenize the sentence
    We remove all the stop words defined in the stopwords_universal list, composed by the stop words of every language nltk supports.
'''
def removeStopWordsAndTokenize(sentence):
    return [stemmingAndLemmatize(w) for w in word_tokenize(sentence) if not (w.lower() in stopwords_universal) and not (w.lower() in string.punctuation)]

'''
    This function is responsible for extracting the feature from a tokenized phrase
    We loop through every token in all_words and check if the word is present inside the tokens list passed as argument
    I've found two ways of doing this:
    1)  return dict([(word, word in tokens) for word in all_words])
        With this method we are returning the words present and not present in the tokens list
    2)  return dict([word, True] for word in tokens)
        With this method we are returning just the tokens present in the tokens list
        
    The second method is faster to compute, however both of the methods have the same performance
'''
def extract_features(tokens):
    return dict([(word, word in tokens) for word in all_words])
    #return dict([word, True] for word in tokens)


'''
    This function generates the stopwords_universal list
'''
def generate_stopwords():
    global stopwords_universal
    for lang in stopwords.fileids():
        stopwords_universal += stopwords.words(lang)

if __name__ == "__main__":

    # generating the stopwords list
    generate_stopwords()

    # processing english phrases
    print("Processing english phrases...")
    for language in e_languages:
        sentences = language.sents()[:int(len(language.sents()) / 10)]
        words = []
        for line in tqdm(sentences, total=len(sentences)):
            phrase = " ".join(line)
            words += removeStopWordsAndTokenize(phrase)
            labeled_phrases.append((phrase, "eng"))

        mc = nltk.FreqDist(words).most_common(int(len(words) / 2))
        mc_list = [w for (w, o) in mc]
        all_words += mc_list

    # processing not english phrases
    print("Processing not-english phrases...")
    for language in ne_languages:
        sentences = language.sents()[:int(len(language.sents()) / 30)]
        words = []
        for line in tqdm(sentences, total=len(sentences)):
            phrase = " ".join(line)
            words += removeStopWordsAndTokenize(phrase)
            labeled_phrases.append((phrase, "not-eng"))

        mc = nltk.FreqDist(words).most_common(int(len(words) / 2))
        mc_list = [w for (w, o) in mc]
        all_words += mc_list

    # creating the documents list, containing the tokenized phrase and the label
    documents = [(removeStopWordsAndTokenize(phrase), category)
                 for (phrase, category) in labeled_phrases]

    # shuffling the list of documents
    random.shuffle(documents)

    # generating the features
    print("Generating the features...")
    word_features = list(nltk.FreqDist(w.lower() for w in all_words))
    features_sets = [(extract_features(doc), label) for (doc, label) in documents]

    # generating the train and test set
    threshold = len(features_sets) - int(len(features_sets) / 5)
    train_set, test_set = features_sets[:threshold], features_sets[threshold:]

    # creating and training the classifier
    print("Training the classifier...")
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    # printing the most informative features
    classifier.show_most_informative_features()

    # testing and calculating the performances
    print("Testing the performances...")
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    labels = []
    tests = []

    for i, (words, label) in enumerate(test_set):
        refsets[label].add(i)
        observed = classifier.classify(words)
        testsets[observed].add(i)
        labels.append(label)
        tests.append(observed)

    print("Accuracy: ", nltk.classify.accuracy(classifier, test_set))
    print("Eng precision: ", nltk.precision(refsets['eng'], testsets['eng']))
    print("Not-eng precision: ", nltk.precision(refsets['not-eng'], testsets['not-eng']))
    print("Eng recall: ", nltk.recall(refsets['eng'], testsets['eng']))
    print("Not-eng recall: ", nltk.recall(refsets['not-eng'], testsets['not-eng']))
    print("Confusion matrix:")
    print(nltk.ConfusionMatrix(labels, tests))
