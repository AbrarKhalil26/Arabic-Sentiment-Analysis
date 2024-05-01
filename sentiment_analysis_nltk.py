import collections
import random
import re
from collections import Counter
from itertools import islice
import string
import nltk
from nltk.metrics.scores import f_measure, precision, recall
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

arabic_diacritics = re.compile(""" ّ    | # Tashdid
                    َ    | # Fatha
                    ً    | # Tanwin Fath
                    ُ    | # Damma
                    ٌ    | # Tanwin Damm
                    ِ    | # Kasra
                    ٍ    | # Tanwin Kasr
                    ْ    | # Sukun
                    ـ     # Tatwil/Kashida
                """, re.VERBOSE)


def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text

def remove_repeating_char(text):
    if 'الله' in text:
        return re.sub(r'(.)\1{2,}', r'\1\1', text)  # keep 2 repeats if 'الله' is present
    else:
        return re.sub(r'(.)\1+', r'\1', text)  # keep only 1 repeat otherwise

def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def remove_repeating_spaces(text):
    return re.sub(' +', ' ', text)

def remove_stop_words(text):
    arabic_stop_words = set(stopwords.words('arabic'))
    words = word_tokenize(text)
    return ' '.join([w for w in words if w not in arabic_stop_words])

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def window(words_seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(words_seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def process_text(text, grams=False):
    clean_text = remove_diacritics(text)
    clean_text = remove_repeating_char(clean_text)
    clean_text = remove_punctuations(clean_text)
    clean_text = remove_repeating_spaces(clean_text)
    clean_text = remove_stop_words(clean_text)
    clean_text = normalize_arabic(clean_text)
    if grams is False:
        return clean_text.split()
    else:
        tokens = clean_text.split()
        grams = list(window(tokens))
        grams = [' '.join(g) for g in grams]
        grams = grams + tokens
        return grams

def document_features(document, corpus_features):
    document_words = set(document) # create a set of unique words in the document
    features = {}
    for word in corpus_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

all_features = list()
texts = list()
data_labels = list()

positive_file = 'datasets/positive_tweets_arabic_20181207_300.txt'
negative_file = 'datasets/negative_tweets_arabic_20181207_300.txt'

n_grams_flag = False
min_freq = 13

with open(positive_file, encoding='utf-8') as tweets_file:
    for line in tweets_file:
        text_features = process_text(line, grams=n_grams_flag)
        all_features += text_features
        texts.append(text_features)
        data_labels.append('pos')

# read negative data
with open(negative_file, encoding='utf-8') as tweets_file:
    for line in tweets_file:
        text_features = process_text(line, grams=n_grams_flag)
        all_features += text_features
        texts.append(text_features)
        data_labels.append('neg')

print('data size', len(data_labels))
print('# of positive', data_labels.count('pos'))
print('# of negative', data_labels.count('neg'))

tweets = [(t, l) for t, l in zip(texts, data_labels)]
random.shuffle(tweets)
for t in tweets[:10]: print(t)  # see the first 10 instances
all_features_freq = nltk.FreqDist(w for w in all_features)
thr = min_freq / len(all_features)

# ---------------------------------------------------
# remove features that have frequency below the threshold
my_features = set([word for word in all_features if all_features_freq.freq(word) > thr])
feature_sets = [(document_features(d, my_features), c) for (d, c) in tweets]
train_percentage = 0.8
splitIndex = int(len(tweets) * train_percentage)
random.shuffle(feature_sets)
train_set, test_set = feature_sets[:splitIndex], feature_sets[splitIndex:]
y_train = [l for t, l in train_set]
y_test = [l for t, l in test_set]

print('data set:', Counter(data_labels))
print('train:', Counter(y_train))
print('test:', Counter(y_test))

print('training NaiveBayes classifier ...')
classifier = nltk.NaiveBayesClassifier.train(train_set)
print('training is done')

ref_sets = collections.defaultdict(set)
test_sets = collections.defaultdict(set)

for i, (feats, label) in enumerate(test_set):
    ref_sets[label].add(i)
    observed = classifier.classify(feats)
    test_sets[observed].add(i)

print('accuracy:', nltk.classify.accuracy(classifier, test_set))
print('positive f-score:', f_measure(ref_sets['pos'], test_sets['pos']))
print('negative f-score:', f_measure(ref_sets['neg'], test_sets['neg']))


classifier.show_most_informative_features(20)