import unittest as ut
import pandas as pd
import numpy as np
from main import dataset, dicDay, dicTime, dicMeal, dicGoal, codedDataset, solSpace #Import structures
from main import getAllWildcard #Import functions 

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
    datasetConcat = pd.concat([dataset, codedDataset], axis=1);
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

  def testNumberOfPatterns(self):
    headers = codedDataset.columns;
    contPatterns = 1;
    
    for header in headers[:-1]:
      contPatterns *= (len(codedDataset[header].unique())+1);

    self.assertEqual(contPatterns, len(solSpace), "The Solution Space has not the correct number of patterns");
  
  #Test that the patterns have the correct symbols in its expressions
  def testCorrectSymbols(self):
    res = solSpace.apply(lambda x :( x[0] in dicDay.values() or x[0] == -1) and (x[1] in dicTime.values() or x[1] == -1) and (x[2] in dicMeal.values() or x[2] == -1), axis=1)
    self.assertEqual(res.sum(), res.count());

class testFunctions(ut.TestCase):

  def setUp(self):

    self.solSpace = np.array([
      [ 1,  1,  1, 0],
      [-1, -1, -1, 0],      
      [ 0,  1,  2, 0],
      [-1, -1,  0, 0]
    ]);

  def testGetAllWildcard(self):
    res = getAllWildcard(self.solSpace);
    expectedValue = self.solSpace[1,:-1];
    self.assertEqual((res == expectedValue).all(),True); #All-wildcard correctly returned
    self.assertEqual(self.solSpace[1,-1], 1); #Pattern correctly marked


if __name__ == '__main__':
  ut.main();