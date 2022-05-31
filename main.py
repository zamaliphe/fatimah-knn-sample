from kNNClassifier import kNNClassifier


def readData(fileName):
    '''
        This method is to read the data from a given file
    '''
    data = []
    labels = []

    with open(fileName, "r") as file:
        lines = file.readlines()
    for line in lines:
        splitline = line.strip().split(',')
        data.append(splitline)
        labels.append(splitline[-1])
    return data, labels

# HayesRoth Data
trainFile = 'Datasets/HayesRoth/hayes-roth.data'

trainData, trainLabel = readData(trainFile)

trainFeatures = []
for row in trainData:
    index = row[0:]
    temp = [int(item) for item in index]
    trainFeatures.append(temp)

trainLabels = [int(label) for label in trainLabel]

knn=kNNClassifier()
knn.fit(trainFeatures, trainLabels)


testFile = 'Datasets/HayesRoth/hayes-roth.test'
testData, testLabel = readData(testFile)

testFeatures = []
for row in testData:
    index = row[0:]
    temp = [int(item) for item in index]
    testFeatures.append(temp)

trainData, trainLabel = readData(trainFile)

trainFeatures = []
for row in trainData:
    index = row[0:]
    temp = [int(item) for item in index]
    trainFeatures.append(temp)

trainLabels = [int(label) for label in trainLabel]


# Call the kFCVEvaluate function of kNNClassifier class

print('*'*20)
print('Hayes Roth Data')

knn.kFCVEvaluate(trainFeatures, 10, 3, 'euclidean')

print("Done")
