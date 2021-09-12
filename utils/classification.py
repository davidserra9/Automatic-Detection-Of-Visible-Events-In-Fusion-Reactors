
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, svm
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_validate, GroupKFold
from sklearn.metrics import plot_precision_recall_curve, precision_recall_curve, f1_score, recall_score, precision_score, accuracy_score, classification_report
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import xgboost as xgb
import os



def arrangeData(dataList):

    shape = (len(dataList[0]),len(dataList))
    data =np.zeros(shape)
    i=0
    for element in dataList:

        data[:,i] = element [:]
        i = i + 1

    return data


def scatterPlot(X, Y, xLabel, yLabel, dataClass):

    x0 = X[np.where(dataClass == 0)]
    y0 = Y[np.where(dataClass == 0)]
    x1 = X[np.where(dataClass == 1)]
    y1 = Y[np.where(dataClass == 1)]
    plt.scatter(x0, y0, c='r', label='False Positive')
    plt.scatter(x1, y1, c='g', label='Hot Spot')

    #plt.scatter(X, Y, c=dataClass, label=dataClass)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend()
    plt.show()

def saveModel(clf, fileName = 'classifier.joblib'):
    PROJECT_ROOT = '/content/drive/My Drive/Colab Notebooks/TFG'
    modelName = PROJECT_ROOT + "/models/" + fileName

    dump(clf, modelName)


def loadModel(fileName='classifier.joblib'):
    PROJECT_ROOT = '/content/drive/My Drive/Colab Notebooks/TFG'
    modelName = PROJECT_ROOT + "/models/" + fileName

    clf = load(modelName)
    return clf

class data_classifier():

    def __init__(self, n=5, classifier = "SVM", multiclass = False, timeAnalysis = False, OneVsRest = False, fileName = "classification"):
        self.scaler = preprocessing.StandardScaler()
        self.n = n
        if self.n > 0:
            self.pca = PCA(n_components = n)
        self.classifier = classifier
        self.multiclass = multiclass
        self.timeAnalysis = timeAnalysis
        self.OneVsRest = OneVsRest
        self.fileName = fileName
        return


    def train_classifier_GS(self, x,y, groups):

        if self.classifier == "SVM":
            clf = svm.SVC(kernel='rbf', probability=True)
            if self.OneVsRest:
                param_grid = {
                    'estimator__C': [1, 10, 100, 1000],
                    'estimator__gamma': [0.001, 0.0001]
                }
            else:

                param_grid = {
                    'C': [1, 10, 100, 1000],
                    'gamma': [0.001, 0.0001]
                }

        elif self.classifier == "RF":
            clf = RandomForestClassifier()
            if self.OneVsRest:
                param_grid = {
                    'estimator__n_estimators': [100, 250, 500, 750, 1000],
                    'estimator__max_features': ['auto', 'log2'],
                    'estimator__max_depth': [4, 6, 8],
                    'estimator__criterion': ['gini', 'entropy']
                }

            else:
                param_grid = {
                    'n_estimators': [100, 250, 500, 750, 1000],
                    'max_features': ['auto', 'log2'],
                    'max_depth': [4, 6, 8],
                    'criterion': ['gini', 'entropy']
                }

        elif self.classifier == "XGB":
            clf = xgb.XGBClassifier()
            if self.OneVsRest:
                param_grid = {
                    'estimator__max_depth': [4, 6, 8],
                    'estimator__learning_rate': [0.01, 0.1, 0.3]
                }
            else:
                param_grid = {
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.1, 0.3]
                }

        #Apply PCA transformation to data
        if self.n > 0:
            self.pca.fit_transform(x)
            x_transformed =self.pca.transform(x)
        else:
            x_transformed = x

        # OneVsRest Classifier
        if self.OneVsRest:
            clf = OneVsRestClassifier(clf)

        # Define cross validation method
        #kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)
        group_kfold = GroupKFold(n_splits=5)
        kfolds = group_kfold.split(x_transformed, y, groups)

        #Get scoring metrics
        scoring = self.get_scoring()

        #Perform Grid Search
        grid = GridSearchCV(clf, param_grid=param_grid, cv=kfolds, n_jobs=1,
                            scoring=scoring, refit='f1_weighted', verbose=1, return_train_score=True)
        grid.fit(x_transformed, y)

        #Print and log results
        self.save_results(grid)
        self.save_best_result(grid)

        #Save best estimator
        self.clf = grid.best_estimator_

        #disp = plot_precision_recall_curve(best_clf, x_transformed, y)

        return

    def plot_precision_recall_curve_best_clf(self, x, y):

        #Apply PCA transformation to data
        if self.n > 0:
            self.pca.fit_transform(x)
            x_transformed =self.pca.transform(x)
        else:
            x_transformed = x

        if not self.OneVsRest:
            print("precision recall curve not defined for multiclass classifiers")
            return
        # precision recall curve
        precision = dict()
        recall = dict()
        n_classes = int(max(y))+1
        y_score = self.clf.predict_proba(x_transformed)
        y_one_hot= np.zeros((y.size, int(y.max()+1)))
        y_one_hot[np.arange(y.size), y.astype(int)] = 1
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_one_hot[:, i],
                                                                y_score[:, i])
            #plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
            if i == 0:
                label = "Wrong detection"
            elif i == 1:
                label = "Hot Spot"
            elif i == 2:
                label = "Anomaly"

            plt.plot(recall[i], precision[i], lw=2, label=label)
            plt.xlabel("recall")
            plt.ylabel("precision")
            plt.legend(loc="best")
            plt.title("precision vs. recall curve")
        #plt.show()
        #disp = plot_precision_recall_curve(self.clf, x_transformed, y)
        plotName = self.fileName + '.png'
        plotName = os.path.join("models", plotName)
        PROJECT_ROOT = '/content/drive/My Drive/Colab Notebooks/TFG/'
        plotName = PROJECT_ROOT + 'plotName'
        plt.savefig(plotName)
        plt.clf()

        return





    def get_scoring(self):
        if self.multiclass:
            scoring = ['accuracy',
                       'f1_weighted',
                       'precision_weighted',
                       'recall_weighted',
                       'f1_macro',
                       'precision_macro',
                       'recall_macro',
                       'f1_micro',
                       'precision_micro',
                       'recall_micro',
                       ]
        else:
            scoring = ['accuracy',
                       'f1',
                       'precision',
                       'recall']
        return scoring

    def save_results(self, grid):

        fileName = 'classification_summary.txt'
        logName = os.path.join("models", fileName)
        PROJECT_ROOT = '/content/drive/My Drive/Colab Notebooks/TFG/'
        logName = PROJECT_ROOT +'logName'
        f = open(logName, "a")
        datetime_str = str(datetime.now())


        self.log()
        self.log("============================================================================================================================================================")
        now = datetime.now()
        dt_string=now.strftime("%d/%m/%Y %H:%M:%S")
        self.log("Finsihed classification training Job at {}" .format(dt_string))
        self.log("Classification using {0} model and {1} PCA reduction".format(self.classifier, self.n))
        self.log("Best parameters set found on train set:")
        self.log()
        self.log(grid.best_params_)
        self.log()


        scores = self.get_scoring()

        meanScores = []
        stdScores = []
        meanValScores = []
        stdValScores = []
        headerNames = []

        for score in scores:


            meansName = 'mean_train_{}'.format(score)
            stdName = 'std_train_{}'.format(score)

            means = grid.cv_results_[meansName]
            stds = grid.cv_results_[stdName]

            meansValName = 'mean_test_{}'.format(score)
            stdValName = 'std_test_{}'.format(score)

            meansVal = grid.cv_results_[meansValName]
            stdsVal = grid.cv_results_[stdValName]

            meansValName = 'mean_val_{}'.format(score)
            stdValName = 'std_val_{}'.format(score)

            meanScores.append(means)
            stdScores.append(stds)
            meanValScores.append(meansVal)
            stdValScores.append(stdsVal)
            headerNames.append(meansName)
            headerNames.append(stdName)
            headerNames.append(meansValName)
            headerNames.append(stdValName)

            logIndividualScores = True
            if logIndividualScores:
                self.log()
                self.log("Grid score {} on Train and Validation set:".format(score))
                self.log()
                for mean, std, meanVal, stdVal, params in zip(means, stds, meansVal, stdsVal, grid.cv_results_['params']):
                    self.log("%0.3f (+/-%0.03f) %0.3f (+/-%0.03f) for %r"
                          % (mean, std * 2, meanVal, stdVal, params))

        self.log()
        self.log("Summary Table of results")
        self.log()

        header= ""
        for name in headerNames :
            header = header + name + "; "
        header = header + "Params"
        self.log(header)

        for i in range(len(grid.cv_results_['params'])):
            results = ""
            for j in range(len(meanScores)):
                mean = meanScores[j][i]
                std = stdScores[j][i]
                meanVal = meanValScores[j][i]
                stdVal = stdValScores[j][i]

                results = results + ("%0.3f; (+/-%0.03f); %0.3f; (+/-%0.03f); "
                    % (mean, std, meanVal, stdVal))

            results = results + str(grid.cv_results_['params'][i])
            self.log(results)
            results = self.fileName + "; " + datetime_str + "; " + results
            f.writelines(str(results))
            f.writelines('\n')

        f.writelines('\n')
        f.close()

        self.log()
        self.log("============================================================================================================================================================")
        self.log()

    def save_best_result(self, grid):


        fileName = 'classification_best_summary.txt'
        logName = os.path.join("models", fileName)
        PROJECT_ROOT = '/content/drive/My Drive/Colab Notebooks/TFG/'
        logName = PROJECT_ROOT + logName
        f = open(logName, "a")
        datetime_str = str(datetime.now())

        scores = self.get_scoring()
        i = grid.best_index_
        results =""

        meanScores = []
        stdScores = []
        meanValScores = []
        stdValScores = []


        for score in scores:

            meansName = 'mean_train_{}'.format(score)
            stdName = 'std_train_{}'.format(score)

            means = grid.cv_results_[meansName]
            stds = grid.cv_results_[stdName]

            meansValName = 'mean_test_{}'.format(score)
            stdValName = 'std_test_{}'.format(score)

            meansVal = grid.cv_results_[meansValName]
            stdsVal = grid.cv_results_[stdValName]

            meanScores.append(means)
            stdScores.append(stds)
            meanValScores.append(meansVal)
            stdValScores.append(stdsVal)


        i = grid.best_index_
        for j in range(len(meanScores)):
            mean = meanScores[j][i]
            std = stdScores[j][i]
            meanVal = meanValScores[j][i]
            stdVal = stdValScores[j][i]

            results = results + ("%0.3f; (+/-%0.03f); %0.3f; (+/-%0.03f); "
                                 % (mean, std, meanVal, stdVal))


        results = results + str(grid.cv_results_['params'][i])
        results = self.fileName + "; " + datetime_str + "; " + results

        f.writelines(str(results))
        f.writelines('\n')
        f.close()




    def predictLabel(self, x, y):

        x_transformed= 0
        if self.n > 0:
            x_transformed = self.pca.transform(x)
        else:
            x_transformed = x

        y_predict = self.clf.predict(x_transformed)

        self.log('Test results with best estimator:')
        self.log(classification_report(y, y_predict))

        """"

        accuracy = accuracy_score(y, y_predict)
        f1 = f1_score(y, y_predict)
        recall = recall_score(y, y_predict)
        precision = precision_score(y, y_predict)

        log('Test results with best estimator:')

        log('Test Accuracy: {}'.format(accuracy))

        log('Test F1: {}'.format(f1))

        log('Test Precision: {}'.format(precision))

        log('Test Recall: {}'.format(recall))"""


    def probability(self, x, y):

        # y_predict = self.clf.predict(x) NOT USED
        probability = self.clf.predict_proba(x)
        return probability

    def predict_probability(self, x, y, threshold = 0.5):
        
        self.pca.fit_transform(x)
        x = self.pca.transform(x)
        probability = self.probability(x,y)
        probabilityHS = probability[:,1]
        # y_predict = np.zeros(y.shape) NOT USED

        y_predict = np.where(probabilityHS < threshold, 0, 1)
        accuracy = accuracy_score(y, y_predict)
        f1 = f1_score(y, y_predict)
        recall = recall_score(y, y_predict)
        precision = precision_score(y, y_predict)

        print('Test Accuracy: {}'.format(accuracy))

        print('Test F1: {}'.format(f1))

        print('Test Precision: {}'.format(precision))

        print('Test Recall: {}'.format(recall))

    def normalize_data(self,x, fit = True):
        x = x.transpose()

        if self.timeAnalysis:
            if fit:
                normalized_x = self.scaler.fit_transform(x)
            else:
                normalized_x = self.scaler.transform(x)
        else:

            nBins = 256

            ROIArea = x[:,0]
            ROIWidth = x[:, 1]
            ROIHeight = x[:, 2]
            ROIFormFactor = x[:, 3]
            ROIFillFactor = x[:, 4]
            ROICentroidX = x[:, 5]
            ROICentroidY = x[:, 6]
            ROIMean = x[:, 7]
            ImageMean = x[:, 8]
            aux0 = int(9 + nBins)
            ROIHist = x[:, 9:aux0]
            aux1 = int(aux0 + 10)
            ROIHistLBP = x[:, aux0:aux1]
            aux2 = int(aux1 + nBins+1)
            RingMean = x[:, aux1]
            RingHist = x[:, int(aux1+1):aux2]
            RingMaskedMean = x[:, aux2]

            scaled_x = np.concatenate((x[:, 0:8], RingMean.reshape(-1, 1), RingMaskedMean.reshape(-1, 1)), axis=1)
            if fit:
                scaled_x = self.scaler.fit_transform(scaled_x)
            else:
                scaled_x = self.scaler.transform(scaled_x)

            #scaled_x = x[:, 6:15]
            normalized_x = np.concatenate((scaled_x, ROIHist, ROIHistLBP, RingHist, x[:, (aux2+1):]), axis = 1)

            #return scaled_x
        return normalized_x

    def predict(self, x):

        x = arrangeData([x])
        x = np.nan_to_num(x)
        x = self.normalize_data(x, fit=False)
        x_transformed = self.pca.transform(x)
        result = self.clf.predict(x_transformed)
        return result

    def grid_search (self, x,y):

        #Deprecated Function!

        self.pca.fit_transform(x)

        x_transformed =self.pca.transform(x)

        kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)
        parameters = {'kernel': ('linear', 'rbf')}
        if self.multiclass:
            scoring = ['accuracy','f1_weighted', 'precision_weighted', 'recall_weighted']
            grid = GridSearchCV(svm.SVC(probability=True), param_grid=parameters, cv=kfolds, n_jobs=4,
                                scoring=scoring, refit='accuracy', verbose=1, return_train_score=True)
        else:
            scoring = ['accuracy','f1','precision','recall']
            grid = GridSearchCV(svm.SVC(probability=True), param_grid=parameters, cv=kfolds, n_jobs=4,
                                scoring=scoring, refit = 'accuracy', verbose=1, return_train_score = True)

        grid.fit(x_transformed, y)

        print (grid.cv_results_)

        if self.multiclass:
            self.log('Mean Validation F1 [Linear, rbf]')
            self.log(grid.cv_results_['mean_test_f1_weighted'])
            self.log('Mean Train F1 [Linear, rbf]')
            self.log(grid.cv_results_['mean_train_f1_weighted'])
            self.log('Mean Validation Precision [Linear, rbf]')
            self.log(grid.cv_results_['mean_test_precision_weighted'])
            self.log('Mean Train Precision [Linear, rbf]')
            self.log(grid.cv_results_['mean_train_precision_weighted'])
            self.log('Mean Validation Recall [Linear, rbf]')
            self.log(grid.cv_results_['mean_test_recall_weighted'])
            self.log('Mean Train Recall [Linear, rbf]')
            self.log(grid.cv_results_['mean_train_recall_weighted'])

        else:
            self.log('Mean Validation F1 [Linear, rbf]')
            self.log(grid.cv_results_['mean_test_f1'])
            self.log('Mean Train F1 [Linear, rbf]')
            self.log(grid.cv_results_['mean_train_f1'])
            self.log('Mean Validation Precision [Linear, rbf]')
            self.log(grid.cv_results_['mean_test_precision'])
            self.log('Mean Train Precision [Linear, rbf]')
            self.log(grid.cv_results_['mean_train_precision'])
            self.log('Mean Validation Recall [Linear, rbf]')
            self.log(grid.cv_results_['mean_test_recall'])
            self.log('Mean Train Recall [Linear, rbf]')
            self.log(grid.cv_results_['mean_train_recall'])

        self.log('Mean Validation Accuracy [Linear, rbf]')
        self.log(grid.cv_results_['mean_test_accuracy'])
        self.log('Mean Train Accuracy [Linear, rbf]')
        self.log(grid.cv_results_['mean_train_accuracy'])

        self.clf = grid.best_estimator_

        return

    def cross_validate (self, x,y):

        # Deprecated Function!

        self.pca.fit_transform(x)

        x_transformed =self.pca.transform(x)

        kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)
        clf = svm.SVC(probability=True, kernel = 'linear')
        if self.multiclass:
            scoring = ['accuracy','f1_weighted', 'precision_weighted', 'recall_weighted']
        else:
            scoring = ['accuracy','f1','precision','recall']

        scores = cross_validate(clf, x_transformed, y, scoring = scoring, cv= kfolds, n_jobs = 5, verbose=1, return_train_score = True)

        print (scores)

        if self.multiclass:
            self.log('Mean Validation F1 [Linear, rbf]')
            self.log(np.mean(scores['test_f1_weighted']))
            self.log('Mean Train F1 [Linear, rbf]')
            self.log(np.mean(scores['train_f1_weighted']))
            self.log('Mean Validation Precision [Linear, rbf]')
            self.log(np.mean(scores['test_precision_weighted']))
            self.log('Mean Train Precision [Linear, rbf]')
            self.log(np.mean(scores['train_precision_weighted']))
            self.log('Mean Validation Recall [Linear, rbf]')
            self.log(np.mean(scores['test_recall_weighted']))
            self.log('Mean Train Recall [Linear, rbf]')
            self.log(np.mean(scores['train_recall_weighted']))

        else:
            self.log('Mean Validation F1 [Linear, rbf]')
            self.log(np.mean(scores['test_f1']))
            self.log('Mean Train F1 [Linear, rbf]')
            self.log(np.mean(scores['train_f1']))
            self.log('Mean Validation Precision [Linear, rbf]')
            self.log(np.mean(scores['test_precision']))
            self.log('Mean Train Precision [Linear, rbf]')
            self.log(np.mean(scores['train_precision']))
            self.log('Mean Validation Recall [Linear, rbf]')
            self.log(np.mean(scores['test_recall']))
            self.log('Mean Train Recall [Linear, rbf]')
            self.log(np.mean(scores['train_recall']))

        self.log('Mean Validation Accuracy [Linear, rbf]')
        self.log(np.mean(scores['test_accuracy']))
        self.log('Mean Train Accuracy [Linear, rbf]')
        self.log(np.mean(scores['train_accuracy']))

        return


    def train_classifier_precision_recall_curve (self, x,y):

        # Deprecated Function!

        self.pca.fit_transform(x)

        x_transformed =self.pca.transform(x)


        kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)
        parameters = {'kernel': ('linear','rbf')}
        if self.multiclass:
            scoring = ['accuracy','f1_weighted', 'precision_weighted', 'recall_weighted']
            grid = GridSearchCV(svm.SVC(probability=True), param_grid=parameters, cv=kfolds, n_jobs=1,
                                scoring=scoring, refit='accuracy', verbose=1, return_train_score=True)
        else:
            scoring = ['accuracy','f1','precision','recall']
            grid = GridSearchCV(svm.SVC(probability=True), param_grid=parameters, cv=kfolds, n_jobs=-1,
                                scoring=scoring, refit = 'accuracy', verbose=1, return_train_score = True)

        grid.fit(x_transformed, y)
        #grid.fit(x, y)

        best_clf = grid.best_estimator_
        self.clf = grid.best_estimator_

        disp = plot_precision_recall_curve(best_clf, x_transformed, y)

        return

    def log(self, value=""):
        print(value)
        # fileName = self.fileName + '.txt'
        # logName = os.path.join("models", fileName)
        PROJECT_ROOT = '/content/drive/My Drive/Colab Notebooks/TFG'
        logName = PROJECT_ROOT + 'models' + self.fileName + '.txt'
        f = open(logName, "a")
        f.writelines(str(value))
        f.writelines('\n')
        f.close()




