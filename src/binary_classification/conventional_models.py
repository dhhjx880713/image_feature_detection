import h5py
import numpy as np
import os
import glob
import cv2
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import glob
import random


num_trees = 100
seed      = 9
npc = 100
TRAINING_DATA_PATH = r'../../data/train' 
TEST_DATA_PATH = r'../../data/test'

def create_models():
    models = []
    models.append(('LR', LogisticRegression(random_state=seed)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier(random_state=seed)))
    models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(random_state=seed)))
    return models


def create_training_test_data(target_size=(512, 512)):
    """preprocessing pipeline
       resize
    """
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    for class_id in next(os.walk(TRAINING_DATA_PATH))[1]:
        images = glob.glob(os.path.join(TRAINING_DATA_PATH, class_id, '*'))
        for image in images:
            image_data = cv2.imread(image)
            image_data = cv2.resize(image_data, target_size)
            train_X.append(np.ravel(image_data))
            train_Y.append(int(class_id))
    
    for class_id in next(os.walk(TEST_DATA_PATH))[1]:
        images = glob.glob(os.path.join(TEST_DATA_PATH, class_id, '*'))
        for image in images:
            image_data = cv2.imread(image)
            image_data = cv2.resize(image_data, target_size)
            test_X.append(np.ravel(image_data))
            test_Y.append(int(class_id))
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)

def main():
    models = create_models()
    results = []
    names = []
    accuracy_scores = []
    train_X, train_Y, test_X, test_Y = create_training_test_data()

    ## reduce the dimension of image data
    pca = PCA(n_components=npc)
    pca.fit(train_X)
    train_X = pca.transform(train_X)
    ## random shuffle the data
    n_samples = len(train_X)
    random_index = np.array(range(n_samples))
    random.shuffle(random_index)
    train_X = train_X[random_index]
    train_Y = train_Y[random_index]

    for name, model in models:
        kfold = KFold(n_splits=10, random_state=seed)
        cv_results = cross_val_score(model, train_X, train_Y, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = "{}: {} {}".format(name, cv_results.mean(), cv_results.std())
        print(msg)
    

    ### visual results
    fig = plt.figure()
    fig.suptitle("Machine Learning algorithm comparison")
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_ylabel("Accuracy")
    ax.set_xticklabels(names)
    plt.show()

    ### evaluate in the test data
    test_X = pca.transform(test_X)
    accuracy_scores = {}
    for name, model in models:
        ## fit the model
        model.fit(train_X, train_Y)
        accuracy_scores[name] = model.score(test_X, test_Y)
    
    print(accuracy_scores)
    fig = plt.figure()
    fig.suptitle("Comparison on test data")
    ax = fig.add_subplot(111)
    plt.bar(accuracy_scores.keys(), accuracy_scores.values())
    ax.set_ylabel("Accuracy")
    # ax.set_xticklabels(accuracy_scores.keys())
    plt.show()



if __name__ == "__main__":
    main()


