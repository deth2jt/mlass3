import csv
import sys
import random
import math
import pandas as pd
import random
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
#https://medium.com/30-days-of-machine-learning/day-3-k-nearest-neighbors-and-bias-variance-tradeoff-75f84d515bdb
#https://stats.stackexchange.com/questions/285866/why-does-a-bagged-tree-random-forest-tree-have-higher-bias-than-a-single-decis
#https://www.kaggle.com/sflender/comparing-random-forest-pca-and-knn
#https://github.com/jeesaugustine/handwriting-recognition-using-random-forests/blob/master/Hand%20Writing%20Recognition%20using%20Random%20Forrest%20Classifier%20.ipynb


def getFeatList(filename, feat):
    lst = []
    df = pd.read_csv(filename,   nrows=1)
    columns = list(df.head(0))
    for val in feat:
        lst += [columns.index(val)]
    return lst


def getRecords(filename):
    records = []
    yRec = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            #42001
            #if(line_count == 1):
                #print("row",len(row))
            if(line_count > 0 and line_count < 4000):
                
            #if(line_count > 0 ):
                #print("leng", len(next(csv_reader)) )
                #numoFColums = len(next(csv_reader))
                cell = []
                #if()
                #print("columncolumncolumncolumn",numoFColums)
                #for column in range(numoFColums):
                for column in range(len(row)):
                    #print("row[column]",row[column])
                    cell += [float(row[column])]
                    #if(column == 0):
                       # print("float(row[column])",int(row[column]))
                        
                    yRec += [int(row[column])]
                records += [cell]
            line_count += 1
                    #print("cell", cell)
            #if(line_count % 100 == 0):
                #print("reading line ", line_count)

        #line_count=line_count-1
    return records, yRec

    
def featureExtract(records, intensity):
    arraySize = len(records[0])
    #print("arraySize", arraySize)

    array = []
    for index in range(arraySize):
        array += [0]

    for row in range(len(records)):
        for column in range(arraySize):
            if(records[row][column] != 0):
                array[column] += 1
    imp = []
    for column in range(arraySize):
        if(array[column] > intensity and column > 0):
            imp += [column]
    #print("imp", imp, len(imp))
    return imp


def createFixedSizedArray(features):
    array = []
    for index in range(features):
        array += [0]
    return array

def getSumCount(records, features):
    sumValue = createFixedSizedArray(features)

    for count in range(len(records)):
        for row in range(len(features)):

            cell = records[count][features[row]]
            sumValue[row] += cell

    return sumValue


    



def normalize(records, features):
    low = 0
    high = 255
    for count in range(len(records)):
        #cell = records[count]
        for column in range(len(features)):
        #for column in range(1,784):

            records[count][features[column]] = (records[count][features[column]] - low) / (high-low)
            #records[count][column] = (records[count][column] - low) / (high-low)

        
    return records

def printRecords(records):  
    for count in range(len(records)):
        print(count, ": ", records[count])

def setTrainValidateTestRecords(records, yRec, upperLImit):

    
    line_count= len(records)
    if(upperLImit < line_count):
        line_count = upperLImit
    testValSize = .10*line_count

    #print("line_count", line_count)
    #print("testValSize", int(testValSize))

    testData = []
    validateData = []
    trainData = []

    ytest = []
    yvalidate = []
    ytrain = []

    for count in range(line_count):
        #row = 0
        if(count < testValSize):
            #print("count:", yRec[count])
            testData = testData + [records[count]]
            ytest = ytest + [yRec[count]]
            #row = row + 1
        elif(count < 2*testValSize):
            validateData = validateData + [records[count]]
            yvalidate = yvalidate + [yRec[count]]
        else:
            trainData = trainData + [records[count]]
            ytrain = ytrain + [yRec[count]]

    return(testData,validateData,trainData, ytest,yvalidate,ytrain)
    

def distanceFrom(records, item1, feature):
    item = []
    for count in range(len(records)):
        if(item1[0] == records[count][0]):
            item = records[count]
    value = 0
    for featurelen in (range(len(feature))):
            pos = feature[featurelen]
            distanceMetric = pow(item[pos] - item1[pos], 2)
            value = value + (distanceMetric)
    return value

def knn(records, item, feature, k):
    #print("item", item)
    #print("distance",distanceFrom(records,item, feature))
    distance = []
    for count in range(len(records)):
        value = 0
        
        for featurelen in (range(len(feature))):
            pos = feature[featurelen]
            distanceMetric = (records[count][pos] - item[pos])**2
            value = value + (distanceMetric)
        #value = abs(records[count][1] - item[1] ) **2

        distance = distance + [(value, count, records[count][1])]
    #printRecords(distance)
    #print("records[1]", records[1])
    # print("Distance before sorting: ", distance[:k])

    distance.sort()
    # print("Distance after sorting: ", distance[:k])


    #printRecords(distance[0:k+1])
    distanceList = distance[0:k]
    neightbours = []
    for count in range(len(distanceList)):
        neightbours += [records[distanceList[count][1]]]
    #printRecords(neightbours)
    return neightbours



def voting(listOfNeighbours, item, feature):
    voteDict = dict()
    for count in range(len(listOfNeighbours)):
        #print("listOfNeighbourslistOfNeighbours", listOfNeighbours[count])
        vote = listOfNeighbours[count][0]
        if(voteDict.has_key(vote)):
            count = voteDict.get(vote)
            voteDict.update( {vote: count+1}  )
        else:
            voteDict.update( {vote: 1}  )

    #majorityPrev = sorted(voteDict.items(), key = lambda kv:(kv[1], kv[0]))[-2][0]  
    majorityPrev = sorted(voteDict.items(), key = lambda kv:(kv[1], kv[0]))[-1][0]        
    majority = sorted(voteDict.items(), key = lambda kv:(kv[1], kv[0]))[-1][0]

    #print("sorted",sorted(voteDict.items(), key = lambda kv:(kv[1], kv[0])))  

    val1 = voteDict.get(majority)
    val2 = voteDict.get(majorityPrev)
    dist1 = 0
    dist2 = 0

    if(val1 == val2):
        for count in range(len(listOfNeighbours)):
            
            #if(listOfNeighbours[count][0] == majority):
            if(listOfNeighbours[count][0] == majority):
                #dist1 = 
                mipmap=listOfNeighbours[count]
                for featurelen in (range(len(feature))):
                    pos = feature[featurelen]
                    distanceMetric = pow(mipmap[pos] - item[pos],2)
                    dist1 = dist1 + distanceMetric

            if(listOfNeighbours[count][0] == majorityPrev):
                #dist1 = 
                mipmap=listOfNeighbours[count]
                for featurelen in (range(len(feature))):
                    pos = feature[featurelen]
                    distanceMetric = pow(mipmap[pos] - item[pos],2)
                    dist2 = dist2 + distanceMetric

    if(dist1 < dist2):
        majority = majorityPrev
    

    #print("itemitem", item[0])
    #print("majority", majority)
    return majority

def bin(records, digit):
    value = 0
    for count in range(len(records)):
        if( records[count][0] == digit):
            value += 1
    return count 



def accuracy(records, outcomes, confusion):
    value = 0
    #confusion = dict()
    for count in range(len(records)):
        if( records[count][0] == outcomes[count]):
            value += 1
        pair=(outcomes[count], records[count][0])
        if(confusion.has_key(pair)):
            count = confusion.get(pair)
            confusion.update( {pair: count+1}  )
        else:
            confusion.update( {pair: 1}  )
            
    #occurance = 

    return ((value/float(len(records))) * 100, confusion)

if __name__ == '__main__':
    records,y = getRecords('train_m.csv')
    #features = featureExtract(records, 11)
    #[]
    features = ["pixel522", "pixel427", "pixel551" ,"pixel372", "pixel483", "pixel467", "pixel356", "pixel455" ,"pixel400" ,"pixel456",
                "pixel496", "pixel428", "pixel495", "pixel550", "pixel384", "pixel468", "pixel440", "pixel412" , "pixel523"]
    #features = ["pixel522", "pixel427", "pixel551" ]
    featuresList =  getFeatList('train_m.csv', features)
    normalize(records,featuresList)
    #47
    #69
    #146

    random.seed(19)
    random.shuffle(records)

    upperLImit = 50000
    testData, validateData, trainData, ytest,yvalidate,ytrain = setTrainValidateTestRecords(records,y, upperLImit)
    #print("len(testData)", len(trainData),  ytest)

    #RandomForrest
    treeLIst = []
    scoresList = []
    for trees in range(1,8):
        n_estimatorsVal = trees*20
        treeLIst += [n_estimatorsVal]
        classifier = ensemble.RandomForestClassifier(n_estimators = n_estimatorsVal, n_jobs=1, criterion="gini")
        classifier.fit(trainData, ytrain)
        scores = classifier.score(validateData, yvalidate)
        scoresList += [scores]
        print("RF number of 20 trews with gini with dataset of ", upperLImit, " :", scores)

    # Data for plotting
    t = treeLIst
    s = scoresList

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='RF number of tress with gini', ylabel='Accuracy score(%)',
        title='Accuracy score vs RF trees (gini) with dataset of 50000')
    ax.grid()

    #fig.savefig("test.png")
    plt.show()

    treeLIst = []
    scoresList = []

    for trees in range(1,8):
        n_estimatorsVal = trees*40
        classifier = ensemble.RandomForestClassifier(n_estimators = n_estimatorsVal, n_jobs=1, criterion="gini")
        classifier.fit(trainData, ytrain)
        scores = classifier.score(validateData, yvalidate)
        scoresList += [scores]
        treeLIst += [n_estimatorsVal]
        print("RF number of 40 tress with gini with dataset of 50000", scores)


    # Data for plotting
    t = treeLIst
    s = scoresList

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='RF number of tress with gini', ylabel='Accuracy score(%)',
        title='Accuracy score vs RF trees (gini) with dataset of 50000')
    ax.grid()

    #fig.savefig("test.png")
    plt.show()


    #knn
    scoresKNNList = []
    kList = []

    for k in range(1,8):
        
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(trainData, ytrain)
        scores = classifier.score(validateData, yvalidate)
        scoresKNNList += [scores]
        kList += [k]
        print("knn scores for ", k, " neighbours score from 50000 datasamples:", scores)
    
    # Data for plotting
    t = kList
    s = scoresKNNList

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='kNN values', ylabel='Accuracy score(%)',
        title='Accuracy score vs kNN values 50000 datasamples')
    ax.grid()

    #fig.savefig("test.png")
    plt.show()


    upperLImit = 10000
    testData, validateData, trainData, ytest,yvalidate,ytrain = setTrainValidateTestRecords(records,y, upperLImit)
    #print("len(testData)", len(trainData),  ytest)

    #RandomForrest
    treeLIst = []
    scoresList = []
    for trees in range(1,8):
        n_estimatorsVal = trees*20
        treeLIst += [n_estimatorsVal]
        classifier = ensemble.RandomForestClassifier(n_estimators = n_estimatorsVal, n_jobs=1, criterion="gini")
        classifier.fit(trainData, ytrain)
        scores = classifier.score(validateData, yvalidate)
        scoresList += [scores]
        print("RF number of 20tress with gini with dataset of ", upperLImit, " :", scores)

    # Data for plotting
    t = treeLIst
    s = scoresList

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='RF number of tress with gini', ylabel='Accuracy score(%)',
        title='Accuracy score vs RF trees (gini) with dataset of 10000')
    ax.grid()

    #fig.savefig("test.png")
    plt.show()

    treeLIst = []
    scoresList = []

    for trees in range(1,8):
        n_estimatorsVal = trees*40
        treeLIst += [n_estimatorsVal]
        classifier = ensemble.RandomForestClassifier(n_estimators = n_estimatorsVal, n_jobs=1, criterion="gini")
        classifier.fit(trainData, ytrain)
        scores = classifier.score(validateData, yvalidate)
        scoresList += [scores]
        #treeLIst += [trees]
        print("RF number of 40tress with gini with dataset of 10000", scores)


    # Data for plotting
    t = treeLIst
    s = scoresList

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='RF number of tress with gini', ylabel='Accuracy score(%)',
        title='Accuracy score vs RF trees (gini) with dataset of 10000')
    ax.grid()

    #fig.savefig("test.png")
    plt.show()



    treeLIst = []
    scoresList = []

    for trees in range(1,8):
        n_estimatorsVal = trees*40
        treeLIst += [n_estimatorsVal]
        classifier = ensemble.RandomForestClassifier(n_estimators = n_estimatorsVal, n_jobs=1, criterion="entropy")
        classifier.fit(trainData, ytrain)
        scores = classifier.score(validateData, yvalidate)
        scoresList += [scores]
        #treeLIst += [trees]
        print("RF number of 40 tress with entropy with dataset of 10000", scores)


    # Data for plotting
    t = treeLIst
    s = scoresList

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='RF number of tress with entropy', ylabel='Accuracy score(%)',
        title='Accuracy score vs RF trees (entropy) with dataset of 10000')
    ax.grid()

    #fig.savefig("test.png")
    plt.show()


    #knn
    scoresKNNList = []
    kList = []

    for k in range(1,8):
        n_estimatorsVal = trees*100
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(trainData, ytrain)
        scores = classifier.score(validateData, yvalidate)
        scoresKNNList += [scores]
        kList += [k]
        print("knn scores for ", k, " neighbours score from 10000 datasamples:", scores)
    
    # Data for plotting
    t = kList
    s = scoresKNNList

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='kNN values', ylabel='Accuracy score(%)',
        title='Accuracy score vs kNN values 10000 datasamples')
    ax.grid()

    #fig.savefig("test.png")
    plt.show()

    #knnCount = 11
    '''
    learnedOutcomes = []
    #print("len(testData)", len(testData) )
    #confused = dict()


    for index in range(0,len(testData)):
    #for index in range(1):
        classOfData = voting(knn(trainData, testData[index], features, knnCount), testData[index], features)
        #print("testData[1]",testData[1])
        #print("class",classOfData)
        learnedOutcomes += [classOfData]
        if(index % 100 == 0):
            print("index",index)
    print("learnedOutcomes",learnedOutcomes)
    print("accuracy",accuracy(testData, learnedOutcomes,confused)[0])
    print("confusion",accuracy(testData, learnedOutcomes, confused)[1])
    
    #printRecords(records)

    #sum = getSumCount(records)[0]
    #countMat = getSumCount(records)[1]
    '''