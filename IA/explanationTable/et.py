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

#t tuple of size n from the dataset
#p pattern of size n from the explanation table
#match(t,p)=1 if they match
def match(t,p):
  n = len(t);
  for i in range(n):
    if not((t[i] == p[i]) or (p[i] == -1)):
      return False;
  return True;

#t tuple of size n from the dataset
#P set of patterns
#fNum(t,P) returns the number of matches of t in P
def fNum(t,P):
  countMatch = 0;
  for p in P:
    countMatch += match(t,p);
  return countMatch;

#Exp(p,D) computes the empirical expectancy of the pattern p in the dataset D
def E(p,D):
  expectancy = 0;
  n = len(p);
  for d in D:
    expectancy += d[-1]*match(d[:n],p);
  return expectancy;  

#t tuple to be estimated
#P set of n patterns
#lambdas set of n optimal lambdas for the optimization problem
#P[i] has the associated lambda[i] value 
def logistic(t,lambdas,P):
  nR,_ = P.shape;
  exponent = 0;
  n = len(P[0]);
  for i in range(nR):
    exponent += lambdas[i]*match(t[:n],P[i]);
  return np.exp(exponent)/(1+np.exp(exponent));

#g function for the Newton's Method algorithm
#lmbd is the value that it's beeing iterated
#p pattern associated with lmbd
#D dataset
#U set o actual estimations
#P set of patterns
def g(lmbd,p,D,U,P):
  nR,_ = D.shape;
  term1 = 0;  
  for i in range(nR):
    term1 += D[i,-2]*U[i]*match(D[i,:-3],p)*np.exp(lmbd*fNum(D[i,:-3],P))
  term2 = -Esp(p, D[:,-3], D[:,:-3]);#E(p,D);
  return term1+term2;

#gPrime derivate function for the Newton's Method algorithm
#lmbd is the value that it's beeing iterated
#p pattern associated with lmbd
#D dataset
#U set o actual estimations
#P set of patterns
def gPrime(lmbd,p,D,U,P):
  nR,_ = D.shape;
  res = 0;

  for i in range(nR):
    res += D[i,-2]*U[i]*match(D[i,:-3],p)*fNum(D[i,:-3],P)*np.exp(lmbd*fNum(D[i,:-3],P))

  return res;    

def newtonMethod(lmbd,p,D,U,P, tau=1e-5, steps=300):
  counterSteps = 0;
  while(np.abs(g(lmbd,p,D,U,P)) > tau and counterSteps<steps):
    lmbd = lmbd - g(lmbd,p,D,U,P)/gPrime(lmbd,p,D,U,P);
    counterSteps+=1;
  return lmbd;

def newtonMethod2(lmbd,p,D,U,P, tau=1e-5, steps=300):
  nR,nC = D.shape;
  grades = np.zeros(nR, dtype=np.int8);
  for i in range(nR):
    grades[i] = fNum(D[i,:-3],P);
  
  coefs = np.zeros(len(np.unique(grades))+1);
  sizePolinom = len(coefs)-1;
  for i in range(nR):
    coefs[sizePolinom-grades[i]] += D[i,-2]*U[i]*match(D[i,:-3],p);
  coefs[sizePolinom] = -Esp(p, D[:,-3], D[:,:-3]);#-E(p,D);
  roots = np.roots(coefs);
  if len(roots) > 0:
    sol = np.log(np.max(roots));
    return sol;
  else:
    return lmbd;
#Si la tupla t coincide con el patron p devuelvo Verdadero
def coincide(t, p):
  for i in range(len(t)):
    if not(p[i] == -1 or t[i] == p[i]):
      return False
  return True

#E(p) esperanza del patron p
def Esp(p, v, D):
  valor=0
  cantCoincidencias=0
  contador=0
  for d in D:
    if match(d, p):
        valor+=v[contador]
        cantCoincidencias+=1
    contador+=1
  E=0
  if cantCoincidencias:
    E=valor/cantCoincidencias
  if not E:
    return 0.01
  return E

 #Probabilidad de las estimaciones
def pr(lambdas):
    num=np.exp(sum(lambdas))
    return num/(1+num)
    
def iterativeScaling2(u, D, T):
  u_aux=np.array(u)
  
  (FD, CD)=np.shape(D[:,:-3])
  (FT, CT)=np.shape(T)

  lambdas=np.zeros(FT)

  #Matriz indicatriz para cada patron
  t_match_p=np.zeros((FT, FD))
  for i in range(FT):
    for j in range(FD):   
      t_match_p[i,j]=match(D[j,:-3], T[i])
      
  #Con cuantos patrones t coincide 
  f_match_t=np.zeros(FD)
  for i in range(FD):
    for j in range(FT):  
      f_match_t[i]+=match(D[i,:-3], T[j])

  #terminos del polinomio
  maximV=int(np.amax(f_match_t))

  for tirada in range(100):
    terminos=np.zeros((FT, maximV+1))
    #Calcular terminos del polinomio
    for i in range(FT):
      for j in range(FD):
        terminos[i, int(f_match_t[j])] += t_match_p[i,j]*u_aux[j]/FD
    #Reacomodar terminos para que terminos[i, 0] sea el coeficiente de mayor grado
    # y terminos[i,-1] sea el termino independiente
    for i in range(FT):
      terminos[i, 0]= -Esp(T[i], D[:,-3], D[:,:-3])
      terminos[i]=np.flip(terminos[i])
      terminos[np.isneginf(terminos)] = 0   
      terminos[np.isinf(terminos)] = 0   
      terminos[np.isnan(terminos)] = 0   

    deltas=np.zeros(FT)
    #Calcular las raices para todos los patrones p  
    #cada delta es la primera raiz no negativa que se encuentre
    for i in range(FT):
      raices=np.roots(terminos[i])
      for raiz in raices:
        if not np.iscomplex(raiz) and raiz > 0:
          deltas[i]=raiz
          break
    deltas=np.log(deltas)
    deltas[np.isneginf(deltas)] = 0   
    deltas[np.isinf(deltas)] = 0   
    deltas[np.isnan(deltas)] = 0   
    lambdas+=deltas
   
    #Calculo de la nueva estimacion
    u_aux_aux=np.array(u_aux)
    for i in range(len(u_aux)):
      lambdasL=np.zeros(FD)
      contador=0
      for p in T:
        #if coincide(D[i, :-1], p):
        if match(D[i,:-3], p):
          lambdasL[contador]=lambdas[contador]      
        contador+=1
      #u_aux_aux[i]=pr(lambdasL)
      #print(pr(lambdasL))
      u_aux_aux[i] = logistic(D[i,:-3],lambdas,T)
      #print(logistic(D[i],lambdas,T))
      #input()

    #Si la nueva estimacion es cercana a la anterior terminar, sino actualizar la estimacion
    if np.allclose(u_aux, u_aux_aux, rtol=1e-05, atol=1e-05, equal_nan=False):
      break
    else:
      u_aux[:]=u_aux_aux[:]

  return u_aux, lambdas

  
def iterativeScaling(T,D,U,lambdas):
  nLambdas = len(lambdas);  
  nR,_= D.shape;  
  newLambdas = np.array(lambdas);
  actU = np.zeros(len(U));
  actU[:] = U[:];

  counter = 0;

  while(True):
    newU = np.zeros(nR);
    aux = [];
    for i in range(nLambdas):
      #aux.append(newtonMethod(newLambdas[i],T[i],D,U,T));
      aux.append(newLambdas[i] + newtonMethod2(newLambdas[i],T[i],D,U,T));
    newLambdas = np.array(aux);
    for i in range(nR):
      newU[i] = logistic(D[i,:-3],newLambdas,T);
    if np.allclose(newU, actU, rtol=1e-05, atol=1e-05, equal_nan=False) or counter<100:
      break
    else:
      actU = np.array(newU)
      counter += 1;
  return newLambdas, newU;

def logLH(T,D,lambdas):
  lLH = 0;
  counter = 0;
  for d in D:
    lLH += d[-1]*np.log(logistic(d,lambdas,T));
    counter += 1;
  return lLH;


def divergenceKL(U,V):
  dKL = 0;
  n = len(V);
  for i in range(n):
    if V[i] and U[i] > 0.0001:
      dKL += np.log(1/U[i])
    elif  (1-U[i]) > 0.0001 :
      dKL += np.log(1/(1-U[i]));
  return dKL;

def gainFunction(T,D,lambdas):
  term1 = 0;
  term2 = 0;
  nR,_ = D.shape;
  rowLambdas,columnLambdas = T.shape;
  
  for i in range(nR):
    exponent = 0;
    for j in range(rowLambdas):
      exponent += lambdas[j]*match(D[i,:columnLambdas],T[j]);
    factor1 = np.log(1+np.exp(exponent));
    term1 += factor1*D[i,-2];

  for i in range(rowLambdas):
    term2 += lambdas[i]*E(T[i], D);
  return -term1+term2;
  
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
  
  #Masks
  listMasks = np.zeros(len(solSpace[:,-1]))

  for i in range(k):
    mask = np.where(solSpace[:,-1] == 0); #Get the unmarked patterns
    logLHArray = np.zeros(len(mask[0]));#Preparing the array to collect the gains
    print(len(mask[0]))
    for index, pCandidate in enumerate(solSpace[mask][:,:-1]): #For each candidate
      #Optimization problem
      TCandidate = np.append(T,[pCandidate], axis=0); #Go to the optimization problem with the actual chosen patterns and with the candidate pCandidate
      auxU = np.array(U);
      lambdasCandidate, newAuxU = iterativeScaling(TCandidate, D, auxU, np.append(lambdas, [0]));#Optimize engineer!
      #newAuxU, lambdasCandidate = iterativeScaling2(auxU, D, TCandidate);#Optimize engineer!
      #logLHArray[index] = gainFunction(TCandidate, D, lambdasCandidate) - pastLogLH; #Compute the gain for these lambdas 
      logLHArray[index] = divergenceKL(newAuxU, D[:,-3]) - pastLogLH #Compute the gain for these lambdas 
      print(index, end='\r');
    indexGain = np.argmin(logLHArray); #Obtain the index gain
    moreGen = -1;
    countWildCards = -1;

    print(solSpace[mask][logLHArray == logLHArray[indexGain]])    
    for indx in np.where(logLHArray == logLHArray[indexGain])[0]:
      actWildcards = 0;
      for val in solSpace[mask][indx]:
        if val == -1:
          actWildcards+=1;
      if countWildCards < actWildcards:
        countWildCards = actWildcards;
        moreGen = indx;

    indexGain = indx;

    chosenPattern = solSpace[mask][indexGain][:-1];
    print('chosen pattern', solSpace[indexGain][:-1])

    T = np.append(T, [chosenPattern], axis=0); #Add the new pattern to the Explanation Table
    for i, sol in enumerate(solSpace):
      if (sol[:-1] == chosenPattern).all():
        solSpace[i] = 1;
        break;

    pastLogLH = logLHArray[indexGain] + pastLogLH; #Update the current gain
    lambdas, U = iterativeScaling(T, D, U, np.append(lambdas, [0]));#Update lambda and the estimation
    #U, lambdas = iterativeScaling2(U, D, T);#Optimize engineer!

  countColumn = [];
  countFraction = [];
  for t in T:
    count = 0;
    fraction = 0;
    for d in D:
      count += match(d[:-3], t)
      fraction += match(d[:-3], t)*d[-3];
    countColumn.append(count);
    countFraction.append(fraction/count);
  T = np.append(np.append(T, np.array(countColumn).reshape(-1,1), axis=1), np.array(countFraction).reshape(-1,1),axis=1);
  return T,U;    

auxSolSpace = [];
for sol in solSpace:
  countMatch = 0;
  for d in codedDataset:
    countMatch += match(d[:-3], sol[:-1]);
  if countMatch > 0:
    auxSolSpace.append(sol);

solSpace = np.array(auxSolSpace);
    
if __name__ == '__main__':
  T,U = explanationTable(codedDataset,3);
  print(T)
  print(U)
  print(divergenceKL(U, codedDataset[:,-3]));