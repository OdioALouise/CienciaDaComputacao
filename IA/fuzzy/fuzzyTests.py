import unittest as ut;
import numpy as np;
#Modifier functions
from fuzzyInference import verySet;
#R functions
from fuzzyInference import Rm, Rs, Rb;
#Composition functions
from fuzzyInference import max_min;

#Test R functions
class RTest(ut.TestCase):
  #Initial setup
  def setUp(self):
    self.u = [0.0,0.5,1.0];
    self.v = [0.0,0.5,1.0];
    
    self.expectedRm = np.array([
      [1.0,1.0,1.0],
      [0.5,0.5,0.5],
      [0.0,0.5,1.0]
    ]);
    
    self.expectedRs = np.array([
      [1.0,1.0,1.0],
      [0.0,1.0,1.0],
      [0.0,0.0,1.0]
    ]);

    self.expectedRb = np.array([
      [1.0,1.0,1.0],
      [0.5,0.5,1.0],
      [0.0,0.5,1.0]
    ]);
    
  #Test RM
  def test_rm(self):
    for i in range(len(self.u)):
      for j in range(len(self.v)):
        with self.subTest(i=i, j=j):
          self.assertAlmostEqual(Rm(self.u[i], self.v[j]), self.expectedRm[i,j], places=6);
    
  #Test RS
  def test_rs(self):
    for i in range(len(self.u)):
      for j in range(len(self.v)):
        with self.subTest(i=i, j=j):
          self.assertAlmostEqual(Rs(self.u[i], self.v[j]), self.expectedRs[i,j], places=6);

  #Test RB
  def test_rb(self):
    for i in range(len(self.u)):
      for j in range(len(self.v)):
        with self.subTest(i=i, j=j):
          self.assertAlmostEqual(Rb(self.u[i], self.v[j]), self.expectedRb[i,j], places=6);


#Test Composition function
class CompositionTest(ut.TestCase):
  def setUp(self):
    self.u = [0.0,0.5,1.0];
    self.v = [0.0,0.5,1.0];
    
  def test_maxMin(self):
    minMaxExpectedValues = np.array([0.25,0.5,1.0]);

    for i in range(len(self.u)):
      with self.subTest(i=i):
        returnedVal, _ = max_min(self.v[i], self.u, Rb, modF=verySet)
        self.assertAlmostEqual(returnedVal, minMaxExpectedValues[i], places=5);


if __name__ == '__main__':
  ut.main();
