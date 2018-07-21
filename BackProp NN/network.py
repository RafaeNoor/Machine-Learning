import numpy as np
import cv2
import pickle
import random

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        print("Loading file {}.pkl ...".format(name))
        obj = pickle.load(f)
        print("Done loading...\n")
        return obj
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        print("Writing to {}.pkl".format(name))
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        print("Done writing...")


def sigmoid(weights, inputs):
    dotProd = weights.dot(inputs)
    if dotProd < 0.0:
        return 1 - 1.0 / (1.0 + np.exp(dotProd))
    else:
        return 1.0 / (1 + np.exp(-1.0 * dotProd))

def normalise(inpArr):
    length = len(inpArr)
    mean = np.mean(inpArr)
    stdDev = np.std(inpArr)
    normalised = (inpArr - np.repeat(mean, length)) * (1.0 / stdDev)
    return normalised

def mseCost(inp,netObj,layerInd, nodeInd,label):
    (output,internals) = netObj.feedForward(inp)
    if(layerInd == 1):
        errorTerm = (-2.0/len(output))*(label[nodeInd]-output[nodeInd])*output[nodeInd]*(1.0-output[nodeInd])
        netErr = np.multiply(errorTerm,internals[int(layerInd)])
        
        return netErr 
    
    elif(layerInd == 0):
        multArray = internals[layerInd]
        weightArr = []
        for i in range(0,len(netObj.network[1])):
            weightArr.append(netObj.network[1][i].weights[nodeInd])
        weightArr = np.array(weightArr)
        ones = np.repeat(1,len(output))
        sumArr =(-2.0/len(output))*(label-output)*output*(ones-output)*weightArr 
        sumArr = np.sum(sumArr)

        netErr = np.multiply(sumArr*multArray[nodeInd]*(1.0-multArray[nodeInd]),inp)
        return netErr
    


class Node:
    def __init__(self, numInputs):
        self.weights = np.random.uniform(-1.0,1.0,numInputs)
    def getWeights(self):
        return self.weights
    def updateWeights(self,newWeights):
        self.weights = newWeights
    def feedForward(self,inputs):
        return sigmoid(self.weights,inputs) 
    def getInfo(self):
        return len(self.weights)


    
class NeuralNetwork:
    def __init__(self, dimArray):
        network = []
        for i in range(1,len(dimArray)):
            layer = []
            for j in range(0,dimArray[i]):
                layer.append(Node(dimArray[i-1]))
            network.append(layer)
        self.network = network

    def printNet(self):
        for layer in self.network:
            print("{} nodes which take {} inputs".format(len(layer),layer[0].getInfo()))
        print("")
    def feedForward(self,inputs):
        if(self.network[0][0].getInfo() != len(inputs)):
            print("ERROR: Input not of correct dimensions")
            return []
        else:
            intermediate = []
            inputs = normalise(inputs)
            intermediate.append(inputs)

            for layer in self.network:
                resultant = []
                for node in layer:
                    resultant.append(node.feedForward(inputs))
                intermediate.append(resultant)
                inputs = resultant
            return (inputs,intermediate) # end result, list of intermediates
    def backProp(self,inp,label,learningRate):
        for i in range(0,len(self.network)):
            j = (len(self.network)-1) - i
            for k in range(0,len(self.network[j])): # for the kth node
                dW = mseCost(inp,self,j,k,label) 
                self.network[j][k].weights =  self.network[j][k].weights +np.multiply(-learningRate,dW)
    
    def measureAccuracy(self,inp,label):
        result = self.feedForward(inp)[0]
        
        if(np.argmax(result) == np.argmax(label)):
            return 1
        else:
            return 0
    def saveWeights(self):
        weightList = []
        for layer in self.network:
            for node in layer:
                weightList.append(node.weights)
        save_obj(weightList,'netWeights')


    def readWeights(self, filename):
        weightList = load_obj('netWeights')
        for i in range(0,len(self.network)): # for each layer
            for j in range(0,len(self.network[i])): # for each node in ith layer
                self.network[i][j].weights = weightList.pop(0)
        print("Done Reading Weights")


        


myNet = NeuralNetwork([1024,150,102])
myNet.printNet()






dataSet = load_obj('dict')
dataSet.pop("101_ObjectCategories",None)

#dataSet = {'cat':[np.random.uniform(0.0,1.0,1024)]}
labelList = []

trainingSet = []
for key in dataSet:
    labelList.append(key)
    label = np.repeat(0.0,102)
    label[labelList.index(key)] = 1.0
    for arr in dataSet[key]:
        trainingSet.append((arr,label))

random.shuffle(trainingSet)

print("Output classes = {}".format(len(dataSet)))
print("Total number of data points = {}".format(len(trainingSet))) #9145

for cycle in range(0,100):
    chunkList = []
    pointsPerChunk = 500 
    ls = []

    for i in range(0,len(trainingSet)):
        if(len(ls) == pointsPerChunk):
            chunkList.append(ls)
            ls = []
        ls.append(trainingSet[i])
    chunkList.append(ls)

    for chunk in chunkList:
        print ("Cycle Number: {}\n=========================================".format(cycle+1))
        total = len(chunk)
        hits = 0
        for point in chunk:
            hits += myNet.measureAccuracy(point[0],point[1])
        print("Pre: Percentage success = {}%".format(hits*100.0/total))

        for epoch in range(0,2):
            print("Epoch: {}".format(epoch+1))
            for point in chunk:
                myNet.backProp(point[0],point[1],0.0005)
        hits = 0
        for point in chunk:
            hits += myNet.measureAccuracy(point[0],point[1])
        print("Post: Percentage success = {}%".format(hits*100.0/total))
        myNet.saveWeights()














        

