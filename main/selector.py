import GWO as gwo
import csv
import numpy
import time
from sklearn.model_selection import train_test_split
import pandas as pd
import fitnessFUNs
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.preprocessing import LabelEncoder

def selector(algo, func_details, popSize, Iter, completeData):
    function_name = func_details[0]
    lb = func_details[1]
    ub = func_details[2]

    DatasetSplitRatio = 0.3  # Training 70%, Testing 30%

    DataFile = completeData

    data_set = numpy.loadtxt(open(DataFile, "rb"), delimiter=",", skiprows=0)
    numRowsData = numpy.shape(data_set)[0]  # number of instances in the  dataset
    numFeaturesData = numpy.shape(data_set)[1] - 1  # number of features in the  dataset

    dataInput = data_set[:, :-1]  # Features
    dataTarget = data_set[:, -1]   # Labels

    # Convert labels to discrete classes using label encoding
    label_encoder = LabelEncoder()
    dataTarget = label_encoder.fit_transform(dataTarget)

    trainInput, testInput, trainOutput, testOutput = train_test_split(dataInput, dataTarget, test_size=DatasetSplitRatio, random_state=1)
    ratio = 0.5
    validationInput, testInput, validationOutput, testOutput = train_test_split(testInput, testOutput, test_size=ratio, random_state=1)

    dim = numFeaturesData

    x = gwo.GWO(getattr(fitnessFUNs, function_name), lb, ub, dim, popSize, Iter, trainInput, trainOutput)

    reducedfeatures = [index for index in range(dim) if x.bestIndividual[index] == 1]
    reduced_data_train_global = trainInput[:, reducedfeatures]
    reduced_data_test_global = testInput[:, reducedfeatures]
    reduced_data_validation_global = validationInput[:, reducedfeatures]

    svc = svm.SVC(kernel='rbf').fit(reduced_data_train_global, trainOutput)

    # Compute the accuracy of the prediction

    target_pred_train = svc.predict(reduced_data_train_global)
    acc_train = float(accuracy_score(trainOutput, target_pred_train))
    x.trainAcc = acc_train

    target_pred_test = svc.predict(reduced_data_test_global)
    acc_test = float(accuracy_score(testOutput, target_pred_test))
    x.testAcc = acc_test

    target_pred_validation = svc.predict(reduced_data_validation_global)
    acc_val = float(accuracy_score(validationOutput, target_pred_validation))
    x.valAcc = acc_val

    return x
