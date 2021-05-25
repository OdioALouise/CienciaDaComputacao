import numpy as np
import pandas as pd
import itertools

#rows - Rows from which obtain the keys of the returned dictionary [dic]
def addKeyValue(rows):
  dic = {};
  uniqueList = rows.unique();
  for val,key in enumerate(uniqueList):
    dic.update({key : val});
  return dic;

def createSolutionSpace(D):
 columns = D.columns;
 _,nC = D.shape;
 dicUniqueCols = {};
 for k, col in enumerate(columns[:-1]):
   dicUniqueCols.update({k : np.append(D[col].unique(),-1)});
 dicUniqueCols.update({nC:[0]}) 
 return pd.DataFrame(np.array(list(itertools.product(*dicUniqueCols.values()))));

#Times of apparitions of x in the dataset
def countApparitions(D,x):
  nRows,_= D.shape;
  res = [True for x in range(nRows)];
  for header in D.columns:
    res &= D[header] == x[header]
  return res.sum()/nRows;

#Calculate the empirical distribution for x in the dataset D
def empirical(D, label):
  res = pd.DataFrame({
    label : D.apply(lambda x:countApparitions(D,x), axis=1)
  });
  return res;


#Data
day = ['Fri', 'Fri', 'Sun', 'Sun', 'Mon', 'Mon', 'Tue', 'Wed', 'Thu', 'Sat', 'Sat', 'Sat', 'Sat', 'Sat'];
time = ['Dawn', 'Night', 'Dusk', 'Morning', 'Afternoon', 'Midday', 'Morning', 'Night', 'Dawn', 'Afternoon', 'Dawn', 'Dawn', 'Dusk', 'Midday'];
meal = ['Banana', 'Green Salad', 'Oatmeal', 'Banana', 'Oatmeal', 'Banana', 'Green Salad', 'Burgers', 'Oatmeal', 'Nuts', 'Banana', 'Oatmeal', 'Rice', 'Toast'];

#Dataset
dataset = pd.DataFrame({
  'day'  : day,
  'time' : time,
  'meal' : meal
}); 

print(dataset)

#Binary data
goal = ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'No', 'No', 'No'];
#Augmented dataset 
dataset['goal'] = goal;

print(dataset)

#Creating dictionaries
dicDay  = addKeyValue(dataset['day']);
dicTime = addKeyValue(dataset['time']);
dicMeal = addKeyValue(dataset['meal']);
dicGoal = addKeyValue(dataset['goal']);
#Coding table
codedDay   = dataset['day'].apply(lambda x: dicDay[x]);
codedTime  = dataset['time'].apply(lambda x: dicTime[x]);
codedMeal  = dataset['meal'].apply(lambda x: dicMeal[x]);
codedGoal  = dataset['goal'].apply(lambda x: dicGoal[x]);
#Coded dataset
codedDataset = pd.DataFrame({
  'day'  : codedDay,
  'time' : codedTime,
  'meal' : codedMeal,
  'goal' : 1-codedGoal
});
print(codedDataset)

solSpace = createSolutionSpace(codedDataset);
print(solSpace)

#Previously - Necessary steps to compute Iterative Scaling
#Extend the codedDataset
###Add empirical probability p(x) to the patterns x
metaColumns = -1; #columns in D that are not part of the pattern, dedicated for calculations
res = empirical(codedDataset.iloc[:,:metaColumns], 'empX'); #calculate empirical distribution
codedDataset = pd.concat([codedDataset,res], axis=1); #add empirical column to the dataset
metaColumns -= 1 #update metacolumns
###Add joint probability p(x,v) to the patterns x with goal value v
res = empirical(codedDataset.iloc[:, :metaColumns+1], 'empXY');
codedDataset = pd.concat([codedDataset,res], axis=1); #add empirical joint probability column to the dataset
metaColumns -= 1 #update metacolumns

print(codedDataset)

#From pandas to numpy
codedDataset = codedDataset.to_numpy();
solSpace = solSpace.to_numpy();

#Get the all-wildcard pattern from 
def getAllWildcard(solSpace):
  _,nC = solSpace.shape;
  patterns = solSpace[:,:-1]; #Get the pattern columns
  arraySum = np.sum(patterns, axis=1) #Sum all the values for each pattern
  index = np.where(arraySum == -(nC-1))[0] #Get the index where the all=wildcard resides
  solSpace[index, -1] = 1; #Mark the pattern as added
  return solSpace[index,:-1][0] #Return the pattern
  
def iterativeScaling(T,D,lambdas):
  #Mock
  nLambdas = len(lambdas);  
  nR,_=D.shape;
  mockLambdas = np.random.uniform(-10,10,nLambdas);
  mockEstimation = np.random.uniform(0,1,nR);
  return mockLambdas, mockEstimation;

def gainFunction(T,D,lambdas):
  #Mock
  mockGain = np.random.uniform(-4,0,1);
  return mockGain;
  
def explanationTable(D,k):
  #Step 1 - Get the all-wildcard pattern and add it to T
  p = getAllWildcard(solSpace)
  T = np.array([p]);
  #Step 2 - Initiate the estimation U 
  nR,nC = D.shape;
  initialEstimation = np.sum(D[:,-3])/nR;
  U = np.array([initialEstimation for _ in range(nR)]);
  lambdas = [0] #Oh yes! This will be a lagrangian relaxation
  pastLogLH = 0;#Best past gain
  for i in range(k):
    mask = np.where(solSpace[:,-1] == 0); #Get the unmarked patterns
    logLHArray = np.zeros(len(mask[0]));#Preparing the array to collect the gains
    for index, pCandidate in enumerate(solSpace[mask][:,:-1]): #For each candidate
      #Optimization problem
      TCandidate = np.append(T,[pCandidate], axis=0); #Go to the optimization problem with the actual chosen patterns and with the candidate pCandidate
      lambdasCandidate, _ = iterativeScaling(TCandidate, D, np.append(lambdas, [0]));#Optimize engineer!
      logLHArray[index] = gainFunction(TCandidate, D, lambdasCandidate) - pastLogLH; #Compute the gain for these lambdas 
    indexGain = np.argmax(logLHArray); #Obtain the index gain
    T = np.append(T, [solSpace[indexGain][:-1]], axis=0); #Add the new pattern to the Explanation Table
    solSpace[indexGain,-1] = 1; #Mark the new added pattern
    pastLogLH = logLHArray[indexGain] + pastLogLH; #Update the current gain
    lambdas, U = iterativeScaling(T, D, np.append(lambdas, [0]));#Update lambda and the estimation
  return T,U;    
    
if __name__ == '__main__':
  explanationTable(codedDataset,3);
