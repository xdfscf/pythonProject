import csv
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.stats as stats
def to_lower_case(s):
    """ Convert a string to lowercase. E.g., 'BaNaNa' becomes 'banana'.
    """
    return s.lower()


def strip_non_alpha(s):
    """ Remove non-alphabetic characters from the beginning and end of a string.

    E.g. ',1what?!"' should become 'what'. Non-alphabetic characters in the middle
    of the string should not be removed. E.g. "haven't" should remain unaltered."""

    s = s.strip()
    if len(s) == 0:
        return s
    if not s[0].isalpha():
        return strip_non_alpha(s[1:])
    elif not s[-1].isalpha():
        return strip_non_alpha(s[:-1])
    else:
        return s


def clean(s):
    """ Create a "clean" version of a string
    """
    return to_lower_case(strip_non_alpha(s))


# Directory of text files to be processed

directory = 'SentenceCorpus/labeled_articles/'

# Learn the vocabulary of words in the corpus
# as well as the categories of labels used per text

categories = {}
vocabulary = {}

num_files = 0
for filename in [x for x in os.listdir(directory) if ".txt" in x]:
    num_files += 1
    print("Processing", filename, "...", end="")
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        with open(f, 'r') as fp:
            for line in fp:
                line = line.strip()
                if "###" in line:
                    continue
                if "--" in line:
                    label, words = line.split("--")
                    words = [clean(word) for word in words.split()]
                else:
                    words = line.split()
                    label = words[0]
                    words = [clean(word) for word in words[1:]]

                if label not in categories:
                    index = len(categories)
                    categories[label] = index

                for word in words:
                    if word not in vocabulary:
                        index = len(vocabulary)
                        vocabulary[word] = index
    print(" done")

n_words = len(vocabulary)
n_cats = len(categories)

print("Read %d files containing %d words and %d categories" % (num_files, len(vocabulary), len(categories)))

print(categories)

# Convert sentences into a "bag of words" representation.
# For example, "to be or not to be" is represented as
# a vector with length equal to the vocabulary size,
# with the value 2 at the indices corresponding to "to" and "be",
# value 1 at the indices corresponding to "or" and "not"
# and zero everywhere else.


X = []
y = []

for filename in [x for x in os.listdir(directory) if ".txt" in x]:
    print("Converting", filename, "...", end="")
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        with open(f, 'r') as fp:
            for line in fp:
                line = line.strip()
                if "###" in line:
                    continue
                if "--" in line:
                    label, words = line.split("--")
                    words = [clean(word) for word in words.split()]
                else:
                    words = line.split()
                    label = words[0]
                    words = [clean(word) for word in words[1:]]

                y.append(categories[label])

                features = n_words * [0]

                bag = {}
                for word in words:
                    if word not in bag:
                        bag[word] = 1
                    else:
                        bag[word] += 1

                for word in bag:
                    features[vocabulary[word]] = bag[word]

                X.append(features)
    print("done")

def train_test(test_size=0.2):




    alpha = pow(2, -15)

    optm_alpha = 0
    optm_accuracy = 0
    all_lower_limit=[]
    all_upper_limit=[]
    all_alpha=[]
    all_avg_acc=[]
    while alpha < pow(2, 5):
        all_acc=[]
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            clf = MultinomialNB(alpha=alpha)
            clf.fit(X_train, y_train)

            para=clf.feature_log_prob_
            predict = clf.predict(X_test)
            np_predict = np.array(predict)
            np_y_test = np.array(y_test)
            tp = np.sum((np_predict == 1) & (np_y_test == 1))

            # True Negative (TN)
            tn = np.sum((np_predict == 0) & (np_y_test == 0))

            # False Positive (FP)
            fp = np.sum((np_predict == 1) & (np_y_test == 0))

            # False Negative (FN)
            fn = np.sum((np_predict == 0) & (np_y_test == 1))

            accuracy = (tp + tn) / len(np_y_test)
            all_acc.append(accuracy)
        avg_acc=np.mean(all_acc)
        std_acc=np.std(all_acc)
        confidence=0.95
        t_score = np.abs(stats.t.ppf((1 - confidence) / 2, 9))
        lower_limit = avg_acc - t_score * (std_acc / np.sqrt(10))
        upper_limit = avg_acc + t_score * (std_acc/ np.sqrt(10))
        all_avg_acc.append(avg_acc)
        all_lower_limit.append(lower_limit)
        all_upper_limit.append(upper_limit)
        all_alpha.append(alpha)
        if avg_acc> optm_accuracy:
            optm_accuracy = avg_acc
            optm_alpha = alpha

        alpha +=0.25
    plt.plot(all_alpha, all_avg_acc)
    plt.fill_between(all_alpha, all_lower_limit, all_upper_limit,color="b",alpha=0.1)
    plt.title('average accuracy with confidence interval')
    plt.xlabel('alpha')
    plt.ylabel('average accuracy')
    plt.legend()
    plt.show()
    print(optm_alpha, optm_accuracy)

clf = MultinomialNB(alpha=10.250030517578125)
clf.fit(X, y)
para=clf.feature_log_prob_

def get_key_from_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    # If the target value is not found, you may handle it accordingly (return None, raise an exception, etc.)
    return None

for i in para:
    sorted_indices = np.argsort(i)

    # Select the top 5 maximum values by taking the last 5 indices
    top_5_max_indices = sorted_indices[-5:]
    for j in top_5_max_indices:
        print(get_key_from_value(vocabulary, j))


train_test(0.2)
