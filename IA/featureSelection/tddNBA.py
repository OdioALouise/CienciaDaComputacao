import unittest as ut;
import numpy as np;
#Structures
from featureSelectionNBA import dicOppId, dicGL, globalMarker;
#Functions
from featureSelectionNBA import typeOfFeatureError, class1Feature, class2Feature, class3Feature, class4Feature, class5Feature, class6Feature;
from featureSelectionNBA import empiricalPXY, fNumeral, g, gPrime, newtonMethod, logLikeliHood;

class testFeatures(ut.TestCase):
  #Simple test cases for features of class 1
  def test_class1Feature(self):
    self.assertEqual(class1Feature([], 1, typeOfFeature='Win') , 1);
    self.assertEqual(class1Feature([], 0, typeOfFeature='Win') , 0);
    self.assertEqual(class1Feature([], 1, typeOfFeature='Lose'), 0);
    self.assertEqual(class1Feature([], 0, typeOfFeature='Lose'), 1);

    #Test the error
    with self.assertRaises(typeOfFeatureError) as cm:
      class1Feature([], 0, typeOfFeature='Lost');
    
    the_exception = cm.exception;
    self.assertEqual(the_exception.__str__(), 'No matching feature');

  #Simple test cases for features of class 2
  def test_class2Feature(self):
    self.assertEqual(class2Feature([-40.0,0,0], 1, typeOfFeature='LoseNormalAndWin'), 1);
    self.assertEqual(class2Feature([-40.0,0,0], 0, typeOfFeature='LoseNormalAndWin'), 0);
    self.assertEqual(class2Feature([ 40.0,0,0], 1, typeOfFeature='LoseNormalAndWin'), 0);

    self.assertEqual(class2Feature([-40.0,0,0], 0, typeOfFeature='LoseNormalAndLose'), 1);
    self.assertEqual(class2Feature([-40.0,0,0], 1, typeOfFeature='LoseNormalAndLose'), 0);
    self.assertEqual(class2Feature([ 40.0,0,0], 0, typeOfFeature='LoseNormalAndLose'), 0);

    self.assertEqual(class2Feature([40.0,0,0], 1, typeOfFeature='WinNormalAndWin'), 1);
    self.assertEqual(class2Feature([40.0,0,0], 0, typeOfFeature='WinNormalAndWin'), 0);
    self.assertEqual(class2Feature([-40.0,0,0], 1, typeOfFeature='WinNormalAndWin'), 0);

    self.assertEqual(class2Feature([40.0,0,0], 0, typeOfFeature='WinNormalAndLose'), 1);
    self.assertEqual(class2Feature([40.0,0,0], 1, typeOfFeature='WinNormalAndLose'), 0);
    self.assertEqual(class2Feature([-40.0,0,0], 0, typeOfFeature='WinNormalAndLose'), 0);

    self.assertEqual(class2Feature([40.0,0,0], 0, typeOfFeature='CloseToWin'), 1);
    self.assertEqual(class2Feature([40.0,0,0], 1, typeOfFeature='CloseToWin'), 1);    
    self.assertEqual(class2Feature([-40.0,0,0],0, typeOfFeature='CloseToWin'), 0);
    self.assertEqual(class2Feature([-40.0,0,0],1, typeOfFeature='CloseToWin'), 0);
    
    self.assertEqual(class2Feature([40.0,0,0], 0, typeOfFeature='CloseToLose'), 0);
    self.assertEqual(class2Feature([40.0,0,0], 1, typeOfFeature='CloseToLose'), 0);    
    self.assertEqual(class2Feature([-40.0,0,0],0, typeOfFeature='CloseToLose'), 1);
    self.assertEqual(class2Feature([-40.0,0,0],1, typeOfFeature='CloseToLose'), 1);

    #Test the error
    with self.assertRaises(typeOfFeatureError) as cm:
      class2Feature([-40.0,0,0], 1, typeOfFeature='WinWinCondition');
    
    the_exception = cm.exception;
    self.assertEqual(the_exception.__str__(), 'No matching feature');

  #Simple test cases for features of class 3
  def test_class3Feature(self):
    self.assertEqual(class3Feature([0,0,0], 1, typeOfFeature='H-W'), 1);
    self.assertEqual(class3Feature([0,0,0], 0, typeOfFeature='H-W'), 0);
    self.assertEqual(class3Feature([0,1,0], 1, typeOfFeature='H-W'), 0);

    self.assertEqual(class3Feature([0,0,0], 0, typeOfFeature='H-L'), 1);
    self.assertEqual(class3Feature([0,0,0], 1, typeOfFeature='H-L'), 0);
    self.assertEqual(class3Feature([0,1,0], 0, typeOfFeature='H-L'), 0);

    self.assertEqual(class3Feature([0,1,0], 1, typeOfFeature='A-W'), 1);
    self.assertEqual(class3Feature([0,1,0], 0, typeOfFeature='A-W'), 0);
    self.assertEqual(class3Feature([0,0,0], 1, typeOfFeature='A-W'), 0);

    self.assertEqual(class3Feature([0,1,0], 0, typeOfFeature='A-L'), 1);
    self.assertEqual(class3Feature([0,1,0], 1, typeOfFeature='A-L'), 0);
    self.assertEqual(class3Feature([0,0,0], 0, typeOfFeature='A-L'), 0);

    self.assertEqual(class3Feature([0,0,0], 0, typeOfFeature='H'), 1);
    self.assertEqual(class3Feature([0,1,0], 1, typeOfFeature='H'), 0);
    self.assertEqual(class3Feature([0,1,0], 0, typeOfFeature='H'), 0);

    self.assertEqual(class3Feature([0,1,0], 0, typeOfFeature='A'), 1);
    self.assertEqual(class3Feature([0,0,0], 1, typeOfFeature='A'), 0);
    self.assertEqual(class3Feature([0,0,0], 0, typeOfFeature='A'), 0);


    #Test the error
    with self.assertRaises(typeOfFeatureError) as cm:
      class3Feature([0,0,0], 1, typeOfFeature='HW and SW');
    
    the_exception = cm.exception;
    self.assertEqual(the_exception.__str__(), 'No matching feature');

  #Simple test cases for features of class 4
  def test_class4Feature(self):
    #Test exception with number outside range
    with self.assertRaises(typeOfFeatureError) as cm:
      class4Feature([0,0,0], 1, teamId=84, result='W');

    the_exception = cm.exception;
    self.assertRegex(the_exception.__str__(), '^Team\s+ID');

    #Test valid cases
    self.assertEqual(class4Feature([0,0, 0],1,teamId=25), 0); #No matching teamId
    self.assertEqual(class4Feature([0,0,25],1,teamId=25, result='W'), 1);
    self.assertEqual(class4Feature([0,0,25],0,teamId=25, result='W'), 0);
    self.assertEqual(class4Feature([0,0,25],0,teamId=25, result='L'), 1);
    self.assertEqual(class4Feature([0,0,25],1,teamId=25, result='L'), 0);
    self.assertEqual(class4Feature([0,0,25],1,teamId=25, result='-'), 1);
    self.assertEqual(class4Feature([0,0,25],0,teamId=25, result='-'), 1);
    self.assertEqual(class4Feature([0,0,23],1,teamId=25, result='-'), 0);


    with self.assertRaises(typeOfFeatureError) as cm:
      class4Feature([0,0,25], 1, teamId=25, result='Deuce');
    
    the_exception = cm.exception;
    self.assertRegex(the_exception.__str__(), '^No\s+matching\s+feature');

  #Simple test cases for features of class 5
  def test_class5Feature(self):
  
    self.assertEqual( class5Feature([0,0,5],1,typeOfFeature='H-W', teamId=5, result='W'),1);
    self.assertEqual( class5Feature([0,0,5],1,typeOfFeature='H-W', teamId=5, result='L'),0);
    self.assertEqual( class5Feature([0,0,5],1,typeOfFeature='H-W', teamId=25, result='W'),0);
    self.assertEqual( class5Feature([0,1,5],1,typeOfFeature='H-W', teamId=5, result='W'),0);

    with self.assertRaises(typeOfFeatureError) as cm:
      class5Feature([0,0,5],1, teamId=-15);
    
    the_exception = cm.exception;
    self.assertRegex(the_exception.__str__(), '^Team\s+ID');

  #Simple test cases for features of class 6
  def test_class6Feature(self):
    
    #Test the exception
    with self.assertRaises(typeOfFeatureError) as cm:
      class6Feature([0,0,0], 1, typeOfFeature='LOL');
  
    the_exception = cm.exception;
    self.assertRegex(the_exception.__str__(), '^No\s+matching\s+feature');
   
    #Test valid inputs
    self.assertEqual(class6Feature([0,0,22],1,typeOfFeature='HW'), 1);
    self.assertEqual(class6Feature([0,0,22],0,typeOfFeature='HW'), 0);    
    self.assertEqual(class6Feature([0,1,22],1,typeOfFeature='HW'), 0);
    self.assertEqual(class6Feature([0,0, 2],1,typeOfFeature='HW'), 0);

    self.assertEqual(class6Feature([0,0,15],0,typeOfFeature='HL'), 1);
    self.assertEqual(class6Feature([0,1,15],0,typeOfFeature='HL'), 0);    
    self.assertEqual(class6Feature([0,0,15],1,typeOfFeature='HL'), 0);
    self.assertEqual(class6Feature([0,0,14],0,typeOfFeature='HL'), 0);
    
    self.assertEqual(class6Feature([0,1,18],1,typeOfFeature='AW'), 1);
    self.assertEqual(class6Feature([0,0,18],1,typeOfFeature='AW'), 0);    
    self.assertEqual(class6Feature([0,1,18],0,typeOfFeature='AW'), 0);
    self.assertEqual(class6Feature([0,1,16],1,typeOfFeature='AW'), 0);
    
    self.assertEqual(class6Feature([0,1,0],0,typeOfFeature='AL'), 1);
    self.assertEqual(class6Feature([0,1,0],1,typeOfFeature='AL'), 0);    
    self.assertEqual(class6Feature([0,0,0],0,typeOfFeature='AL'), 0);
    self.assertEqual(class6Feature([0,1,2],0,typeOfFeature='AL'), 0);

class testProbabilities(ut.TestCase):
  
  def setUp(self):
    #Dataset 1
    self.dataSet1 = np.array([
      [1,1,1,1],
      [2,2,2,0],
      [1,1,1,1]
    ]);
    self.expectedEmpiricalPXY1 = np.array([2.0/3.0, 1.0/3.0, 2.0/3.0]);
    self.expectedEmpiricalPX1 = np.array([2.0/3.0, 1.0/3.0, 2.0/3.0]);
    f1ds1 = lambda X,y : (X == [1,1,1]).all();
    f2ds1 = lambda X,y : (y == 1);
    self.FDS1 = [f1ds1, f2ds1];

    #Dataset 2
    self.dataSet2 = np.array([
      [1,1,1,1],
      [2,2,2,0],
    ]);
    self.expectedEmpiricalPXY2 = np.array([1.0/2.0, 1.0/2.0]);
    self.expectedEmpiricalPX2 = np.array([1.0/2.0, 1.0/2.0]);
    f2ds2 = lambda X,y : y == 0;
    
    self.FDS2 = [f1ds1,f2ds2];

  def test_empiricalPXY(self):
    result1 = empiricalPXY(self.dataSet1);
    result2 = empiricalPXY(self.dataSet2);
    
    for i in range(len(result1)):
      with self.subTest(i=i):
        self.assertAlmostEqual(result1[i], self.expectedEmpiricalPXY1[i], places=5);

    for i in range(len(result2)):
      with self.subTest(i=i):
        self.assertAlmostEqual(result2[i], self.expectedEmpiricalPXY2[i], places=5);

  def test_fNumeral(self):
    #Dataset1 
    ds1ExpectedValues = [2,0,2];
    
    for i in range(len(self.dataSet1)):
      with self.subTest(i=i):
        self.assertEqual(fNumeral(self.dataSet1[i,:-1], self.dataSet1[i,-1], self.FDS1), ds1ExpectedValues[i]);     
    
    #Dataset2
    ds2ExpectedValues = [1,1];

    for i in range(len(self.dataSet2)):
      with self.subTest(i=i):
        self.assertEqual(fNumeral(self.dataSet2[i,:-1], self.dataSet2[i,-1], self.FDS2), ds2ExpectedValues[i]);

  def test_g(self):
    Delta = 1;
    #Dataset1
    pSArray = np.array([[1.0/3.0], [1.0/3.0], [1.0/3.0]]);
    self.dataSet1 = np.append(np.append(np.append(self.dataSet1, self.expectedEmpiricalPXY1.reshape(-1,1), axis=1), self.expectedEmpiricalPX1.reshape(-1,1), axis=1),pSArray,axis=1);
    globalMarker = [1,1,0];
    expectedResult1 = self.dataSet1[0,-2]*self.dataSet1[0,-1]*self.FDS1[0](self.dataSet1[0,:-4],self.dataSet1[0:-4])*np.exp(Delta*fNumeral(self.dataSet1[0,:-4], self.dataSet1[0,-4], self.FDS1)) - self.dataSet1[0,-3]*self.FDS1[0](self.dataSet1[0,:-4],self.dataSet1[0:-4]); 

    self.assertAlmostEqual(g(Delta,self.FDS1[0], self.FDS1, self.dataSet1,testMode=True, testMarker=globalMarker), expectedResult1, places=5);

    #Dataset2
    pSArray = np.array([[1.0/2.0], [1.0/2.0]]);
    self.dataSet2 = np.append(np.append(np.append(self.dataSet2, self.expectedEmpiricalPXY2.reshape(-1,1), axis=1), self.expectedEmpiricalPX2.reshape(-1,1), axis=1),pSArray,axis=1);

    globalMarker = [1,1];
    expectedResult2 = self.dataSet2[0,-2]*self.dataSet2[0,-1]*self.FDS2[0](self.dataSet2[0,:-4], self.dataSet2[0,-4])*np.exp(Delta*fNumeral(self.dataSet2[0,:-4], self.dataSet2[0,-4],self.FDS2)) - self.dataSet2[0,-3]*self.FDS2[0](self.dataSet2[0,:-4], self.dataSet2[0,-4]);
    
    self.assertAlmostEqual(g(Delta, self.FDS2[0], self.FDS2,self.dataSet2, testMode=True, testMarker=globalMarker), expectedResult2, places=5);


  def test_gPrime(self):
    Delta = 1;
    #Dataset1
    pSArray = np.array([[1.0/3.0], [1.0/3.0], [1.0/3.0]]);
    self.dataSet1 = np.append(np.append(np.append(self.dataSet1, self.expectedEmpiricalPXY1.reshape(-1,1), axis=1), self.expectedEmpiricalPX1.reshape(-1,1), axis=1),pSArray,axis=1);
    globalMarker = [1,1,0];
    expectedResult1 = self.dataSet1[0,-2]*self.dataSet1[0,-1]*self.FDS1[0](self.dataSet1[0,:-4],self.dataSet1[0:-4])*fNumeral(self.dataSet1[0,:-4], self.dataSet1[0,-4], self.FDS1)*np.exp(Delta*fNumeral(self.dataSet1[0,:-4], self.dataSet1[0,-4], self.FDS1)); 

    self.assertAlmostEqual(gPrime(Delta,self.FDS1[0], self.FDS1, self.dataSet1,testMode=True, testMarker=globalMarker), expectedResult1, places=5);

    #Dataset2
    pSArray = np.array([[1.0/2.0], [1.0/2.0]]);
    self.dataSet2 = np.append(np.append(np.append(self.dataSet2, self.expectedEmpiricalPXY2.reshape(-1,1), axis=1), self.expectedEmpiricalPX2.reshape(-1,1), axis=1),pSArray,axis=1);

    globalMarker = [1,1];
    expectedResult2 = self.dataSet2[0,-2]*self.dataSet2[0,-1]*self.FDS2[0](self.dataSet2[0,:-4], self.dataSet2[0,-4])*fNumeral(self.dataSet2[0,:-4], self.dataSet2[0,-4],self.FDS2)*np.exp(Delta*fNumeral(self.dataSet2[0,:-4], self.dataSet2[0,-4],self.FDS2));
    
    self.assertAlmostEqual(gPrime(Delta, self.FDS2[0], self.FDS2,self.dataSet2, testMode=True, testMarker=globalMarker), expectedResult2, places=5);

  def test_newtonMethod(self):
    #Ajdust the data to test
    self.dataSet1[1,0]=1;
    self.dataSet1[1,-1]=1;
    self.FDS1[0] = lambda X,y : X[0] == 1;
    pSArray = np.array([[1.0/3.0], [1.0/3.0], [1.0/3.0]]);
    self.dataSet1 = np.append(np.append(np.append(self.dataSet1, self.expectedEmpiricalPXY1.reshape(-1,1), axis=1), self.expectedEmpiricalPX1.reshape(-1,1), axis=1),pSArray,axis=1);
    
    Delta = 0;
    dividend = self.FDS1[0](self.dataSet1[0,:-4], self.dataSet1[0,-4])*self.dataSet1[0,-3] + self.FDS1[0](self.dataSet1[1,:-4], self.dataSet1[1,-4])*self.dataSet1[1,-3];
    divisor = self.dataSet1[0,-2]*self.dataSet1[0,-1]*self.FDS1[0](self.dataSet1[0,:-4],self.dataSet1[0,-4]) + self.dataSet1[1,-2]*self.dataSet1[1,-1]*self.FDS1[0](self.dataSet1[1,:-4], self.dataSet1[1,-4]);
    expectedResult = (1.0/2.0)*np.log((dividend/divisor));
    newtonResult = newtonMethod(Delta, self.FDS1[0], self.FDS1,self.dataSet1, testMode=True, testMarker=[1,1,0]);
    
    self.assertAlmostEqual(expectedResult, newtonResult, places=3);

  def test_logLikelihood(self):
    pSArray = np.array([[1.0/2.0], [1.0/2.0]]);
    self.dataSet2 = np.append(np.append(np.append(self.dataSet2, self.expectedEmpiricalPXY2.reshape(-1,1), axis=1), self.expectedEmpiricalPX2.reshape(-1,1), axis=1),pSArray,axis=1);

    expectedLogLH = -(0.5*np.log(1 + np.exp(0.7))) -(0.5*np.log(1 + np.exp(0.3))) + 0.7*.5 + 0.3*0.5;
    
    self.assertAlmostEqual(expectedLogLH, logLikeliHood(self.FDS2, [0.7,0.3], self.dataSet2),places=4);
    


if __name__ == '__main__':
  ut.main();
