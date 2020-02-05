########################
#
#  172CPG18 YeonJin Jin
#
########################

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

import timeit

print('[Load Data]')

# data processing, CSV file I/O
train = pd.read_csv('C:/YEONJIN/2018-2 GRADUATE/DATA ANALYSIS/PROJECT/input_test/train.csv')
test  = pd.read_csv('C:/YEONJIN/2018-2 GRADUATE/DATA ANALYSIS/PROJECT/input_test/test.csv')



print('[Model Learning]')


#################################
print('1. Load Models & Model Data')
tree = DecisionTreeClassifier(max_depth=10, random_state=0)

#top3 features = [52, 393, 559]
test_features= test.iloc[:,[52, 393, 559]]
train_features = train.iloc[:,[52, 393, 559]]
label = train['Activity']


#################################
print('2. Learning Decesion Tree')

start_time = timeit.default_timer()
fit=tree.fit(train_features,label)
pred=fit.predict(test_features)
elapsed = timeit.default_timer() - start_time
print("Time Elpase: " + str(elapsed))
print("Acc: %3.5f" % (accuracy_score(test['Activity'],pred)))


#################################
print('3. Save Tree Graph')
export_graphviz(tree, out_file="tree.dot", class_names=list(set(label)),
                feature_names=list(set(train_features)),
                impurity=False, filled=True)


#################################

print('4. Confusion Matrix Function Initailize')

import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




#####################################

print('5. Test Model with Simplified Data')

test2  = pd.read_csv('C:/YEONJIN/2018-2 GRADUATE/DATA ANALYSIS/PROJECT/input_model/train.csv')

testData =  test2.iloc[:,[54, 395, 561]]
testLabel = test2.activity.values

from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()

# encoding test labels
encoder.fit(testLabel)
testLabelE = encoder.transform(testLabel)


from sklearn.metrics import precision_score,recall_score,confusion_matrix


y_score = tree.predict_proba(testData)
y_te_pred = tree.predict(testData)
acc = accuracy_score(testLabel, y_te_pred)
print('precision_score')
prec = precision_score(testLabel, y_te_pred, average="macro")
print('recall_score')
rec = recall_score(testLabel, y_te_pred, average="macro")
print('confusion_matrix')
cfs = confusion_matrix(testLabel, y_te_pred)
print("Acc: %3.5f, P: %3.5f, R: %3.5f" % (acc, prec, rec))

# Plot non-normalized confusion matrix
plt.figure()
class_names = encoder.classes_
plot_confusion_matrix(cfs, classes=class_names,
                      title='Confusion Matrix with Decision Tree')
