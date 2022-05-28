import operator

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

