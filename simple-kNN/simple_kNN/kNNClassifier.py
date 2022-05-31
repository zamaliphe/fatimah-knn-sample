import operator
from random import randrange

class kNNClassifier:
    '''
    Description:
        This class contains the functions to calculate distances
    '''
    def __init__(self,k = 3, distanceMetric = 'euclidean'):
        '''
        Description:
            KNearestNeighbors constructor
        Input    
            k: total of neighbors. Defaulted to 3
            distanceMetric: type of distance metric to be used. Defaulted to euclidean distance.
        '''
        pass
    
    def fit(self, xTrain, yTrain):
        '''
        Description:
            Train kNN model with x data
        Input:
            xTrain: training data with coordinates
            yTrain: labels of training data set
        Output:
            None
        '''
        assert len(xTrain) == len(yTrain)
        self.trainData = xTrain
        self.trainLabels = yTrain

    def getNeighbors(self, testRow):
        '''
        Description:
            Train kNN model with x data
        Input:
            testRow: testing data with coordinates
        Output:
            k-nearest neighbors to the test data
        '''

        distances = []
        for i, trainRow in enumerate(self.trainData):
            distances.append([trainRow, self.euclideanDistance(testRow, trainRow), self.trainLabels[i]])
            distances.sort(key=operator.itemgetter(1))

        neighbors = []
        for index in range(self.k):
            neighbors.append(distances[index])
        return neighbors
        
    def predict(self, xTest, k, distanceMetric):
        '''
        Description:
            Apply kNN model on test data
        Input:
            xTest: testing data with coordinates
            k: number of neighbors
            distanceMetric: technique to calculate distance metric
        Output:
            predicted label 
        '''
        self.testData = xTest
        self.k = k
        self.distanceMetric = distanceMetric
        predictions = []
        for i, testCase in enumerate(self.testData):
            neighbors = self.getNeighbors(testCase)
            output= [row[-1] for row in neighbors]
            prediction = max(set(output), key=output.count)
            predictions.append(prediction)
        
        return predictions

    def euclideanDistance(self, vector1, vector2):
        '''
        Description:
            Function to calculate Euclidean Distance

        Inputs:
            vector1, vector2: input vectors for which the distance is to be calculated
        Output:
            Calculated euclidean distance of two vectors
        '''
        vectorA, vectorB = vector1, vector2
        if len(vectorA) != len(vectorB):
            raise ValueError("Undefined for sequences of unequal length.")
        distance = 0.0
        for i in range(len(vectorA) - 1):
            distance += (vectorA[i] - vectorB[i]) ** 2
        return (distance) ** 0.5

    def printMetrics(self, actual, predictions):
        '''
        Description:
            This method calculates the accuracy of predictions
        '''
        assert len(actual) == len(predictions)
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predictions[i]:
                correct += 1
        return (correct / float(len(actual)) * 100.0)

    def crossValSplit(self, dataset, numFolds):
        '''
        Description:
            Function to split the data into number of folds specified
        Input:
            dataset: data that is to be split
            numFolds: integer - number of folds into which the data is to be split
        Output:
            split data
        '''
        dataSplit = list()
        dataCopy = list(dataset)
        foldSize = int(len(dataset) / numFolds)
        for _ in range(numFolds):
            fold = list()
            while len(fold) < foldSize:
                index = randrange(len(dataCopy))
                fold.append(dataCopy.pop(index))
            dataSplit.append(fold)
        return dataSplit

    def kFCVEvaluate(self, dataset, numFolds, *args):
        '''
        Description:
            Driver function for k-Fold cross validation
        '''
        knn = kNNClassifier()
        folds = self.crossValSplit(dataset, numFolds)
        print("\nDistance Metric: ", *args[-1])
        print('\n')
        scores = list()
        for fold in folds:
            trainSet = list(folds)
            trainSet.remove(fold)
            trainSet = sum(trainSet, [])
            testSet = list()
            for row in fold:
                rowCopy = list(row)
                testSet.append(rowCopy)

            trainLabels = [row[-1] for row in trainSet]
            trainSet = [train[:-1] for train in trainSet]
            knn.fit(trainSet, trainLabels)

            actual = [row[-1] for row in testSet]
            testSet = [test[:-1] for test in testSet]

            predicted = knn.predict(testSet, *args)

            accuracy = self.printMetrics(actual, predicted)
            scores.append(accuracy)

        print('*' * 20)
        print('Scores: %s' % scores)
        print('*' * 20)
        print('\nMaximum Accuracy: %3f%%' % max(scores))
        print('\nMean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))





