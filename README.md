## Assignment 1 - Natural Language Processing
### Development in NLTK of a Naive Bayes Classifier able to detect a single class in one of the corpora available by distinguishing english and not-english.

A link to the code can be found [here](https://github.com/DamianoPedoniUNIVR/nlp_assignment_1)

#### - Corpus
I've used the NLTK package and the europarl_raw set of corpus. The classifier is trained with english, german, italian, french and dutch languages.
To simplify the execution, I've used just a tenth of the english set and a thirtieth of each of the others languages sets. The positive class sets is almost 2000 phrases long and the negative class is almost 2800 phrases long.

#### - Train, test and performance
I've used a 1/5 split between training and testing sets.

With this split setup, I've achieved this results:
- Accuracy (how many times the model was correct overall):  0.9897540983606558
- Eng precision (how good the model is at predicting the positive category):  0.9744245524296675
- Not-eng precision (how good the model is at predicting the negative category):  1.0
- Eng recall (how many times the model was able to detect the positive category):  1.0
- Not-eng recall (how many times the model was able to detect the negative category):  0.9831932773109243

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

### - Usage as a probabilistic language model
A Probabilistic Language Model assigns a probability to every sentence in such a way that more likely english sentences get an higher probability. This classifier can be easily used as a Probabilistic Language Model, since we can calculate the percentage of english tokens inside a phrase and so the probability.
