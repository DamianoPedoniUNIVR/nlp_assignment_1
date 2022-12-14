## Assignment 1 - Natural Language Processing
##### Damiano Pedoni
### Development in NLTK of a Naive Bayes Classifier able to detect a single class in one of the corpora available by distinguishing english and not-english.

A link to the code can be found [here](https://github.com/DamianoPedoniUNIVR/nlp_assignment_1), or visiting this url: https://github.com/DamianoPedoniUNIVR/nlp_assignment_1

#### - Corpus
I've used the NLTK package and the europarl_raw set of corpus. The classifier is trained with english, german, italian, french and dutch languages.
To simplify the execution, I've used just a tenth of the english set and a thirtieth of each of the others languages sets. The positive class set is almost 2000 phrases long and the negative class one is almost 2800 phrases long, for a total of almost 15000 words of different languages.

#### - The pipeline
The pipeline of this model is very straight forward:<br>
1) I generate the stopwords list, by taking all the stopwords in all the languages supported by nltk.<br>
2) I process the english corpus. I'm using a tenth of the available sentences just to simplify the execution. Here I ignore the stopwords and tokenize each word in the sentence. After the tokenization process I proceed with stemming and lemmatization. Then I add each token in a list and label each phrase.<br>
3) I process each negative class language, as described above.<br>
4) I process each phrase found as described above, removing stopwords, using tokenization, stemming and lemmatization.<br>
5) I randomize the list to have uniform training and testing sets and to get each time a different result.<br>
6) I generate the features sets and proceed with training and testing.<br>

#### - Train, test and performance
I've used a 5:1 split between training and testing sets, which gives us almost 4000 phrases for training and almost 900 for testing purphoses.

With this split setup, I've achieved this results:<br>
- Accuracy (how many times the model was correct overall):  0.9897540983606558<br>
- Eng precision (how good the model is at predicting the positive category):  0.9744245524296675<br>
- Not-eng precision (how good the model is at predicting the negative category):  1.0<br>
- Eng recall (how many times the model was able to detect the positive category):  1.0<br>
- Not-eng recall (how many times the model was able to detect the negative category):  0.9831932773109243<br>


With this confusion matrix:
```
        |       n |
        |       o |
        |       t |
        |       - |
        |   e   e |
        |   n   n |
        |   g   g |
--------+---------+
    eng |<381>  . |
not-eng |  10<585>|
--------+---------+
(row = reference; col = test)
```
The most informative features found for one of the tests was:
```
state = True                   eng : not-en = 73.6 : 1.0
presid = True                  eng : not-en = 72.4 : 1.0
structur = True                eng : not-en = 64.6 : 1.0
area = True                    eng : not-en = 55.6 : 1.0
system = True                  eng : not-en = 47.6 : 1.0
aid = True                     eng : not-en = 43.1 : 1.0
nation = True                  eng : not-en = 35.2 : 1.0
propos = True                  eng : not-en = 35.0 : 1.0
object = True                  eng : not-en = 34.6 : 1.0
issu = True                    eng : not-en = 32.1 : 1.0
```

#### - Usage as a probabilistic language model
A Probabilistic Language Model assigns a probability to every sentence in such a way that more likely english sentences get an higher probability. This classifier can be easily used as a Probabilistic Language Model, since we can calculate the percentage of english tokens inside a phrase and so obtain the probability.
