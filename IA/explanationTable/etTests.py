import unittest as ut
import pandas as pd
import numpy as np
from et import dataset, dicDay, dicTime, dicMeal, dicGoal, codedDataset, solSpace #Import structures
from et import getAllWildcard, match, fNum, E, logistic, g, gPrime, newtonMethod, gainFunction, logLH, divergenceKL, iterativeScaling#Import functions 
from et import newtonMethod2,iterativeScaling2

class testStructures(ut.TestCase):

  def setUp(self):
    self.entries = 14;
    self.achievedGoals = 7;
    #List of day, time, meal
    self.lDays = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    self.lTime = ['Dawn', 'Morning', 'Midday', 'Afternoon', 'Dusk', 'Night'];
    self.lMeal = ['Banana', 'Oatmeal', 'Nuts', 'Toast', 'Rice', 'Green Salad', 'Burgers'];
    self.lGoal = ['Yes', 'No'];

  def createDataFrame(self, dic, labels):
    return pd.DataFrame(dic, columns=labels);

  def testSizeOfSampleET(self):
    nR,_=dataset.shape;
    self.assertEqual(nR, self.entries, "Incorrect size of dataset rows");

  def testCorrectNamesInColumnTables(self):
    resDay  = dataset['day'].apply(lambda x: x in self.lDays);
    resTime = dataset['time'].apply(lambda x : x in self.lTime);
    resMeal = dataset['meal'].apply(lambda x : x in self.lMeal);
    
    self.assertEqual(self.entries, resDay.sum(), 'Some day has not a correct name');
    self.assertEqual(self.entries, resTime.sum(), 'Some time of the day has not a correct name');
    self.assertEqual(self.entries, resMeal.sum(), 'Some meal has not a correct name'); 

  def testDictionaryNames(self):
    for key in dicDay.keys():
      self.assertEqual(True, key in self.lDays, "The dictionary key is not a correct key for this datasample")
    for key in dicTime.keys():
      self.assertEqual(True, key in self.lTime, "The dictionary key is not a correct key for this datasample")
    for key in dicMeal.keys():
      self.assertEqual(True, key in self.lMeal, "The dictionary key is not a correct key for this datasample")  
    for key in dicGoal.keys():
      self.assertEqual(True, key in self.lGoal, "The dictionary key is not a correct key for this datasample")  

  
  def testDistinctDictionaryValues(self):
    dfDay  = self.createDataFrame(dicDay.values(),  ['Day']);
    dfTime = self.createDataFrame(dicTime.values(), ['Time']);
    dfMeal = self.createDataFrame(dicMeal.values(), ['Meal']);
    dfGoal = self.createDataFrame(dicGoal.values(), ['Goal']);    
    
    self.assertEqual(len(dfDay['Day'].unique()), dfDay['Day'].count(), "Some values in the dictionary are repeated");
    self.assertEqual(len(dfTime['Time'].unique()), dfTime['Time'].count(), "Some values in the dictionary are repeated");
    self.assertEqual(len(dfMeal['Meal'].unique()), dfMeal['Meal'].count(), "Some values in the dictionary are repeated");
    self.assertEqual(len(dfGoal['Goal'].unique()), dfGoal['Goal'].count(), "Some values in the dictionary are repeated");

  def testTableHomomorphism(self):
    auxCodedDataset = pd.DataFrame(data=codedDataset[:,:-2], columns=dataset.columns);
    datasetConcat = pd.concat([dataset, auxCodedDataset], axis=1);
    for i in range(4):
      dic = {};
      if i == 0: #Pick the dictionary
        dic = dicDay;
      elif i == 1:
        dic = dicTime;
      elif i == 2:
        dic = dicMeal;
      elif i == 3:
        dic = dicGoal;
      subDatasetConcat = datasetConcat.iloc[:,[i,i+4]]; #Obtain the relevant columns
      colName = subDatasetConcat.columns[0]; #Rename the second column and create the dataset to test with the renamed column
      testSubDatasetConcat = pd.DataFrame({ 
        colName : subDatasetConcat.iloc[:,0],
        colName+'aux' : subDatasetConcat.iloc[:,1]
      });
      res = testSubDatasetConcat.apply(lambda x:dic[x[0]]==x[1], axis=1); 
      
      if i == 3: #Goal test
        self.assertEqual(self.entries, self.entries - res.sum(), 'Problem no homomophic column {}'.format(colName));
        continue;
      self.assertEqual(self.entries, res.sum(), 'Problem no homomophic column {}'.format(colName));

  def deprecatedtestNumberOfPatterns(self):
    auxCodedDataset = pd.DataFrame(data=codedDataset[:,:-2], columns=dataset.columns);
    headers = auxCodedDataset.columns;
    contPatterns = 1;
    
    for header in headers[:-1]:
      contPatterns *= (len(auxCodedDataset[header].unique())+1);

    self.assertEqual(contPatterns, len(solSpace), "The Solution Space has not the correct number of patterns");

  #Test that the patterns have the correct symbols in its expressions
  def testCorrectSymbols(self):
    auxSolSpace = pd.DataFrame(data=solSpace[:,:-1], columns=dataset.columns[:-1]);
    res = auxSolSpace.apply(lambda x :( x[0] in dicDay.values() or x[0] == -1) and (x[1] in dicTime.values() or x[1] == -1) and (x[2] in dicMeal.values() or x[2] == -1), axis=1)
    self.assertEqual(res.sum(), res.count());

class testFunctions(ut.TestCase):

  def setUp(self):

    self.solSpace = np.array([
      [ 1,  1,  1, 0],
      [-1, -1, -1, 0],      
      [ 0,  1,  2, 0],
      [-1, -1,  0, 0]
    ]);
    
    self.lambdas = np.array([0.5,0.5,0.25,0.25]);

    self.D = np.array([
      [6,0,0,0,0.5,0.5],
      [5,1,3,1,0.5,0.5]
    ]);
    
    self.U = [0.5,0.5];

  def testGetAllWildcard(self):
    res = getAllWildcard(self.solSpace);
    expectedValue = self.solSpace[1,:-1];
    self.assertEqual((res == expectedValue).all(),True); #All-wildcard correctly returned
    self.assertEqual(self.solSpace[1,-1], 1); #Pattern correctly marked

  def testMatchPattern(self):
    self.assertEqual(match(self.D[0,:-3],self.solSpace[0,:-1]), False);
    self.assertEqual(match(self.D[0,:-3],self.solSpace[1,:-1]), True);
    self.assertEqual(match(self.D[0,:-3],self.solSpace[2,:-1]), False);    
    self.assertEqual(match(self.D[0,:-3],self.solSpace[3,:-1]), True);
    self.assertEqual(match(self.D[1,:-3],self.solSpace[0,:-1]), False);
    self.assertEqual(match(self.D[1,:-3],self.solSpace[1,:-1]), True);
    self.assertEqual(match(self.D[1,:-3],self.solSpace[2,:-1]), False);
    self.assertEqual(match(self.D[1,:-3],self.solSpace[3,:-1]), False);    

  def testFNum(self):
    nR,_ = self.D.shape;
    expectedResultTuple1 = 2;
    expectedResultTuple2 = 1;
    arrayExpectedResults = [expectedResultTuple1, expectedResultTuple2];
    for i in range(nR):
      with self.subTest(i=i):
        self.assertEqual(fNum(self.D[i,:-3], self.solSpace[:,:-1]), arrayExpectedResults[i]);

  def testE(self):
    nR,_ = self.solSpace.shape;
    expectedResultTuple1 = 0;
    expectedResultTuple2 = 1;
    expectedResultTuple3 = 0; 
    expectedResultTuple4 = 0.5;    
    arrayExpectedResults = [expectedResultTuple1, expectedResultTuple2, expectedResultTuple3, expectedResultTuple4];
    for i in range(nR):
      with self.subTest(i=i):
        self.assertEqual(E(self.solSpace[i,:-1], self.D), arrayExpectedResults[i]);
    

  def testLogistic(self):
    nR,_ = self.D.shape;
    expectedResultTuple1 = np.exp(3.0/4.0)/(1+np.exp(3.0/4.0));
    expectedResultTuple2 = np.exp(1.0/2.0)/(1+np.exp(1.0/2.0));
    arrayExpectedResults = [expectedResultTuple1, expectedResultTuple2];
    for i in range(nR):
      with self.subTest(i=i):
        self.assertAlmostEqual(logistic(self.D[i,:-3], self.lambdas, self.solSpace), arrayExpectedResults[i], places=4);


  def testG(self):
    nR = len(self.lambdas);
    expectedResultTuple1 = -0.01;
    expectedResultTuple2 = 0.5*0.5*1*np.exp(0.5*2) + 0.5*0.5*1*np.exp(0.5*1) - 0.5;
    expectedResultTuple3 = -0.01; 
    expectedResultTuple4 = (1.0/4.0)*np.exp(1.0/2.0)-0.01;        
    arrayExpectedResults = [expectedResultTuple1, expectedResultTuple2, expectedResultTuple3, expectedResultTuple4];
    for i in range(nR):
      with self.subTest(i=i):
        self.assertAlmostEqual(g(self.lambdas[i],self.solSpace[i,:-1],self.D,self.U,self.solSpace), arrayExpectedResults[i], places=7);

  def testGPrime(self):
    nR = len(self.lambdas);
    expectedResultTuple1 = 0;
    expectedResultTuple2 = (1.0/4.0)*(2*np.exp(1)+np.exp(1.0/2.0));
    expectedResultTuple3 = 0; 
    expectedResultTuple4 = (1.0/4.0)*2*np.exp(1.0/2.0);        
    arrayExpectedResults = [expectedResultTuple1, expectedResultTuple2, expectedResultTuple3, expectedResultTuple4];
    for i in range(nR):
      with self.subTest(i=i):
        self.assertAlmostEqual(gPrime(self.lambdas[i],self.solSpace[i,:-1],self.D,self.U,self.solSpace), arrayExpectedResults[i], places=7);

  def testNewtonMethod(self):
    expectedResult = 0;
    P = np.array([self.solSpace[1,:-1]]);
    self.assertAlmostEqual(newtonMethod(0.5,P[0],self.D,self.U,P), expectedResult, places=4);

  def testGainFunction(self):
    expectedResult = -np.log(1+np.exp(0.5))+0.5;
    self.assertAlmostEqual(gainFunction(self.solSpace[:2,:-1],self.D,self.lambdas[:2]), expectedResult, places=5);
  
  #Test that we are maximizing the logLikelihood
  def testLogLH(self):
    resLH = logLH(self.solSpace[:2,:-1],self.D,self.lambdas[:2]);
    resParametric = gainFunction(self.solSpace[:2,:-1],self.D,self.lambdas[:2]);
    self.assertAlmostEqual(resLH, resParametric, places=7);

  def testDivergenceKL(self):
    expectedRes = 2*np.log(2);
    resKL = divergenceKL(self.U,self.D[:,-3]);
    self.assertAlmostEqual(expectedRes, resKL, places=5);
    
  def testIterativeScaling(self):
    nR,nC = codedDataset.shape;
    initialEstimation = np.sum(codedDataset[:,-3])/nR;
    U = np.array([initialEstimation for _ in range(nR)]);
    T = np.array([
      [-1,-1,-1],
      [6,-1,-1],
      [-1,-1,0],
      [-1,-1,2]
    ]);

    l,u = iterativeScaling(T,codedDataset,U,np.array([0,0,0,0]))
    #u,l = iterativeScaling2(U,codedDataset[:,-3],codedDataset[:,:-3],T);

if __name__ == '__main__':
  ut.main();