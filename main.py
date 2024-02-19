from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn import metrics
from sklearn.preprocessing import OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
def read_txt_data(file_paths):
    # Open the CSV file and create a csv.reader object

    feature_to_num=[]
    label_to_num={}
    with open(file_paths, 'r') as file:
        all_features = []
        all_labels = []
        counter = 0
        for line in file:
            counter += 1
            content = line.strip().split(',')
            if len(feature_to_num)==0:
                feature_to_num = [dict() for i in range(len(content[1:]))]
            features = content[1:]

            label = content[0]

            if len(label_to_num) == 0:
                label_to_num[label] = [0, 0]
            elif label not in label_to_num.keys():
                max_value = len(label_to_num.values())
                label_to_num[label] = [max_value, 0]

            label_to_num[label][1] += 1
            label = label_to_num[label][0]


            for index, feature in enumerate(features):
                if len(feature_to_num[index])==0:
                    feature_to_num[index][feature]=[0, 0, 0]
                elif feature=='?'and feature not in feature_to_num[index].keys():
                    feature_to_num[index][feature] = [np.nan, 0, 0]
                elif feature not in feature_to_num[index].keys():
                    max_value = len(feature_to_num[index])
                    feature_to_num[index][feature]=[max_value, 0, 0]
                features[index] = feature_to_num[index][feature][0]
                if label==0:
                    feature_to_num[index][feature][1] += 1
                else:
                    feature_to_num[index][feature][2] += 1

            all_features.append(features)
            all_labels.append(label)

    return counter, all_labels, all_features, feature_to_num, label_to_num

def train_test(test_size=0.2):
    counter, all_labels, all_features, feature_to_num, label_to_num = read_txt_data("./agaricus-lepiota.data")

    min_categories = [len(features) for features in feature_to_num]
    imp = IterativeImputer(max_iter=10, random_state=0, missing_values=np.nan, min_value=0)
    IterativeImputer(random_state=0)
    imp.fit(all_features)
    all_features = np.round(imp.transform(all_features))
    X_train, X_test, y_train, y_test = train_test_split(all_features,  all_labels, test_size=test_size, random_state=42)

    one_hot_encoding=np.eye(2)
    y_test_one_hot=np.array([one_hot_encoding[i] for i in y_test])

    alpha = pow(2, -15)
    max_auc = 0
    optm_alpha = 0
    optm_accuracy = 0
    optm_f1_score = 0
    optm_fpr = None
    optm_tpr = None
    optm_para=None
    while alpha < pow(2, 5):
        rng = np.random.RandomState(1)
        clf = CategoricalNB(alpha=alpha, min_categories=min_categories)
        clf.fit(X_train, y_train)

        para=clf.feature_log_prob_
        predict_prob_y = clf.predict_proba(X_test)

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
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = (2 * precision * recall) / (precision + recall)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, predict_prob_y[:, 1:], pos_label=1)
        '''

        '''
        test_auc = metrics.roc_auc_score(y_test_one_hot, predict_prob_y)

        if test_auc > max_auc:
            max_auc = test_auc
            optm_alpha = alpha
            optm_accuracy = accuracy
            optm_f1_score = f1_score
            optm_fpr = fpr
            optm_tpr = tpr
            optm_para = para
        alpha *= 1.05
    print(max_auc, optm_alpha, optm_accuracy, optm_f1_score,  optm_para)
    plt.plot(optm_fpr, optm_tpr)
    plt.title('ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.show()

    all_label_one_hot=np.array([one_hot_encoding[i] for i in all_labels])
    rng = np.random.RandomState(1)
    clf = CategoricalNB(alpha=optm_alpha, min_categories=min_categories)
    clf.fit(all_features,  all_labels)
    para = clf.feature_log_prob_
    predict_prob_y = clf.predict_proba(all_features)
    predict = clf.predict(all_features)
    np_predict = np.array(predict)
    np_y_test = np.array(all_labels)
    tp = np.sum((np_predict == 1) & (np_y_test == 1))

    # True Negative (TN)
    tn = np.sum((np_predict == 0) & (np_y_test == 0))

    # False Positive (FP)
    fp = np.sum((np_predict == 1) & (np_y_test == 0))

    # False Negative (FN)
    fn = np.sum((np_predict == 0) & (np_y_test == 1))

    accuracy = (tp + tn) / len(all_labels)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, predict_prob_y[:, 1:], pos_label=1)
    test_auc = metrics.roc_auc_score(all_label_one_hot, predict_prob_y)
    print(test_auc, optm_alpha, accuracy, f1_score, para)
    plt.plot(fpr, tpr)
    plt.title('ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.show()

train_test(0.2)
train_test(0.99)
