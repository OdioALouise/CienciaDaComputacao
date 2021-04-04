import matplotlib.pyplot as plt;
import numpy as np;
import seaborn as sns;
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
iris = sns.load_dataset('iris');

#classical plotting graphs
def my_plotter(ax, data1, data2, param_dict, title=None, xlabel=None, ylabel=None):

  out = ax.plot(data1, data2, **param_dict);
  if title:
    ax.set_title(title);
  if xlabel:
    ax.set_xlabel(xlabel);
  if ylabel:
    ax.set_ylabel(ylabel);

  return out; 


lRate = 0.5;
alpha = 0.0001;
maxIter = 500;

"""
vW = np.array([
  [0,0,1,1,0],
  [0,0,1,1,0],
  [0,0,0,0,1],
  [0,0,0,0,1]
]);

initW = np.array([
  [0.,0.,.5,.3,0.],
  [0.,0.,.4,.2,0.],
  [0.,0.,0.,0.,.7],
  [0.,0.,0.,0.,.15],
]);
"""

#Generate the visibility matrix of weights vW and initialize the matrix of weights W, given the number of input n and training samples p
def initMatrixes(p, n, nH=0):
  h = int(np.round((p-1)/(n+2))); #Recommended hidden layer number
  if nH:
    h = nH;
  vW = np.zeros((h+n, h+n+1));
  W = np.zeros((h+n, h+n+1));
  for i in range(h+n):
    if i < n:
      vW[i, n:n+h] = 1;
      W[i, n:n+h] = np.random.uniform(-1.0/n, 1.0/n, size=h);
      continue;
    vW[i, -1] = 1;
    W[i, -1] = np.random.uniform(-1.0/n, 1.0/n);
  return vW, W, h;

#This function returns
#o[|N|, 2] where |N| is the quantity of neurons and o[:, 0] represents the values
#of the activation functions at time t, and o[:, 1] represents the values of the activation functions
#at time t+1
#
#W[|Weigths|, 3], |Weights| is the amount of weights in the system, and W(t-1) = W[:, 0], W(t)= W[:,1], W(t+1) = W[:, 2] represents the values of the weights at the current time t, the recent past t-1 and the updated time t+1.
#
#
# diccInd[key]=value maps the serialized weights to the visibility matrix vW, so that diccInd[k] = [i,j], being k the index of the weight in the activation array o, and [i,j] the respective connection tho whom the weight attached.

def generateDictAndArrays(vW, W, nI, nH):
  indX = np.where(vW == 1);
  diccInd = {};
  o = np.zeros((nI+nH+1, 1), dtype='float');
  serialW = np.zeros((len(indX[0]), 3));

  for i in range(len(indX[0])):
    diccInd.update({i : (indX[0][i], indX[1][i])});
    serialW[i,0] = W[diccInd[i]];
    serialW[i,1] = W[diccInd[i]];

  return diccInd, o, serialW;


def logistic(u):
  return 1.0/(1.0 + np.power(np.exp(1), -u));



def trainingAlgorithm(dataTraining, vW, W, nI=0, nH=0, passForward=False, species=[]):
  #Init structures
  diccInd, o, serialW = generateDictAndArrays(vW, W, nI, nH);

  learnedNumber = 0;

  infoArray = np.zeros((150, 4))

  for row in dataTraining:
    X = row[:-1];
    y = row[-1];
    
    errO = np.inf;

    counter = 0;

    #oGuesses = [];
    #errSteps = [];
    
    counterMax = 0;

    while(counterMax < maxIter and np.abs(errO) > 1E-2):
      counterMax += 1;
      
      ##############
      #Pass forward#
      #############
      #First the easy: the activation function of the input layer are the inputs
      for i in range(nI):
        o[i] = X[i];

      #Update the activation function array for the hidden layer
      #also put the final activation function (the output layer)
      for i in range(nI, nI+nH+1):
        indX = np.where(vW[:,i] == 1)[0]
        u = 0;
        for j in indX:
          oInd = list(diccInd.keys())[list(diccInd.values()).index((j,i))];
          u += o[j]*serialW[oInd,1];
        if i == nI + nH:
          o[i] = logistic(u + X[-2] + X[-1]);
        else:
          o[i] = logistic(u);

      errO = (y - o[-1])[0];

      if not passForward:
        print("\ry" + str(y) + " - o" + str(o[-1]), counterMax, maxIter, end="");
        #input();

      if passForward:
        print("Predicciones ", y, o[-1], errO, 'Learned number ', species[learnedNumber]);
        diccInd, o, serialW = generateDictAndArrays(vW, W, nI, nH);        
        infoArray[learnedNumber,0] = y;
        infoArray[learnedNumber,1] = y - errO;
        infoArray[learnedNumber,2] = errO;
        infoArray[learnedNumber,3] = species[learnedNumber];        
        break;

      #oGuesses.append(o[-1][0]);
      #errSteps.append(errO);

      #################
      #Backpropagation#
      ################


      #Backpropagate hidden weights
      currW = 0;
      #print(diccInd)
      #print(o);      
      
      #for k in range(len(vW[0]), len(vW[0])-nH, -1):
      for k in range(len(serialW[:,0])-1, len(serialW[:,0])-nH-1, -1):      
        (i,j) = diccInd[k];
        serialW[k,2] = serialW[k,1] + lRate*o[j]*(1-o[j])*errO*o[i] + alpha*(serialW[k,1]-serialW[k,0])
        currW += serialW[k,1];

      errH = errO*currW;

      #Backpropagate input weights
      for k in range(len(serialW[:,0])-nH-1, -1, -1):
        (i,j) = diccInd[k];
        serialW[k,2] = serialW[k,1] + lRate*o[j]*(1-o[j])*errH*o[i] + alpha*(serialW[k,1]-serialW[k,0]);

      #print(errO);
      #print(errH);
      #print(vW);
      #print(X);
      #print(o);
      #print(serialW);
      #print(diccInd);
      #input();
      #Time: t+1 is now t and t is now t-1
      o[nI:] = 0;
      serialW[:,0] = serialW[:,1];
      serialW[:,1] = serialW[:,2];
      serialW[:,2] = 0;

      counter+=1;

      #print(serialW);
      #print(errO);
      #input();
    learnedNumber += 1;
    if not passForward:
      print('Number of iterations ' + str(counter), ' learner number ', learnedNumber);

  if passForward:
    np.save('infoArray',infoArray);
    return;
    
  finalW = np.zeros(vW.shape);
  for k,v in diccInd.items():
    finalW[v]=serialW[k,1];
  return finalW;
    

indexesTrain = np.load('indexesTrain.npy');
indexesTest = np.load('indexesTest.npy');

p = indexesTrain.size; #number of training samples
n = 4;#number of inputs of the network
#initMatrixes(p, n);
print(iris.head());
tam = iris['species'].size;
trainingSet = np.zeros((p,5));
trainingSet[:, 0] = iris['sepal_length'][indexesTrain];
trainingSet[:, 1] = iris['sepal_width'][indexesTrain];
trainingSet[:, 2] = iris['petal_length'][indexesTrain];
trainingSet[:, 3] = iris['petal_width'][indexesTrain];


diccSpecies = {
  'setosa' : 0,
  'versicolor' : 1,
  'virginica' : 0,
}


speciesVal = np.zeros(iris['species'].size);
counter = 0;
for specie in iris['species'][indexesTrain]:
  trainingSet[counter, 4] = diccSpecies[specie]; 
  counter += 1;


for j in range(len(trainingSet[0])-1):
  sumICol = np.sum(trainingSet[:,j]);
  for i in range(len(trainingSet)):
    trainingSet[i,j] = trainingSet[i,j]/sumICol;

print(trainingSet);

####Training test
testSet = np.zeros((indexesTest.size,5));
testSet[:, 0] = iris['sepal_length'][indexesTest];
testSet[:, 1] = iris['sepal_width'][indexesTest];
testSet[:, 2] = iris['petal_length'][indexesTest];
testSet[:, 3] = iris['petal_width'][indexesTest];


speciesVal = np.zeros(iris['species'].size);
counter = 0;
for specie in iris['species'][indexesTest]:
  testSet[counter, 4] = diccSpecies[specie]; 
  counter += 1;


for j in range(len(testSet[0])-1):
  sumICol = np.sum(testSet[:,j]);
  for i in range(len(testSet)):
    testSet[i,j] = testSet[i,j]/sumICol;

print(testSet);
####

dataSet = np.zeros((150,6));
dataSet[:, 0] = iris['sepal_length'];
dataSet[:, 1] = iris['sepal_width'];
dataSet[:, 2] = iris['petal_length'];
dataSet[:, 3] = iris['petal_width'];

diccSpecies = {
  'setosa' : 0,
  'versicolor' : 1,
  'virginica' : 0,
}

diccSpecies_tag = {
  'setosa' : 0,
  'versicolor' : 1,
  'virginica' : 2,
}


speciesVal = np.zeros(iris['species'].size);
counter = 0;
for specie in iris['species']:
  dataSet[counter, 5] = diccSpecies[specie]; 
  dataSet[counter, 4] = diccSpecies_tag[iris['species'][counter]];  
  counter += 1;


for j in range(len(dataSet[0])-2):
  sumICol = np.sum(dataSet[:,j]);
  for i in range(len(dataSet)):
    dataSet[i,j] = (dataSet[i,j]/sumICol)*100;


print(dataSet[:,:-1])
print(dataSet[:,:-1].shape, dataSet[:,-1].shape);
print('dataset');

X_train, X_test, y_train, y_test = train_test_split(dataSet[:,:-1], dataSet[:,-1], stratify=dataSet[:,-1],random_state=1)

print(X_train)
print(y_train)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape);

print(X_train.shape, y_train.shape);
X_test_tags = X_test[:, -1]; 
trainingSet = np.append(X_train[:,:-1], y_train.reshape(-1,1), axis=1);
testSet = np.append(X_test[:,:-1], y_test.reshape(-1,1), axis=1);
print(trainingSet)


vW, W, h = initMatrixes(p, n, nH=100);

np.save('vW',vW);
np.save('W',W);

learnedW = trainingAlgorithm(trainingSet, vW, W, nI=n, nH=h);
np.save('learnedW', learnedW);
trainingAlgorithm(testSet, vW, learnedW, nI=n, nH=h, passForward=True, species=X_test_tags);

