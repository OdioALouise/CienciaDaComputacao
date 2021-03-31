import matplotlib.pyplot as plt
import numpy as np

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


#Dataset x = X[:-1], y = [-1]
X = np.array([0.5,0.3,0.05]);

lRate = 1 #learning rate
alpha = 0.5#balancing momentum

nI=2; #Nodes in input layer
nH=2; #Nodes in hidden layer
nO=1; #Nodes in output layer

#Visibility Weight Matrix
vW = np.array([
  [0,0,1,1,0],
  [0,0,1,1,0],
  [0,0,0,0,1],
  [0,0,0,0,1],
  [0,0,0,0,0],
]);

#Weight Matrix
#W at time t
#Wpast at time t - 1

W = np.array([
  [0.,0.,.5,.3,0.],
  [0.,0.,.4,.2,0.],
  [0.,0.,0.,0.,.7],
  [0.,0.,0.,0.,.15],
  [0.,0.,0.,0.,0.],
]);

Wpast = np.array(W); #np.zeros((nI+nH+nO, nI+nH+nO), dtype='float');

def activationFunction(u):
  return 1.0/(1.0 + np.power(np.exp(1), -u));

#Calculate the ouptut of the network and return the output value, its error with respect to y, and the inner activation function's value
def forwardPass(x, y):
  #Calculate summation function u and activation function o for Hidden layers
  nodes = nI + nH + nO;
  oH = np.zeros(nH); #Activation function values of the hidden nodes 
  for i in range(nI, (nI+nH)):
    idX = np.where(vW[:,i] == 1);
    u = np.sum(W[idX[0], i]*x)
    o = activationFunction(u);
    oH[i-nI] = o; 
  #Calculates the output activation function o
  idX = np.where(vW[:,nodes-1] == 1)[0];
  o = activationFunction(np.sum(W[idX, nodes - 1]*oH));
  return o, y - o, oH;

#Calculate the backward error for all nodes and updates him
def backwardPass(x, ErrO, o, oH):
  #Adjust Hidden node' weights
  for i in range(nI, nI+nH):
    Wfut[i,-1] = W[i,-1] + lRate*(o*(1-o))*ErrO*oH[i-nI] + alpha*(W[i,-1] - Wpast[i,-1]);
  #Calculate the propagated error for the input layers
  indX = np.where(vW[:,-1] == 1)[0];
  ErrI = ErrO*(np.sum(W[indX, -1]));
  for i in range(nI): #Input nodes
    for j in range(nI, nI+nH):
      Wfut[i,j] = W[i,j] + lRate*(oH[j-nI]*(1-oH[j-nI]))*ErrI*x[i] + alpha*(W[i,j]-Wpast[i,j]); 
    
ErrO = np.inf;
counter = 0;
oGuesses = [];
errSteps = [];
while(np.abs(ErrO) > 1E-6 and counter < 1000):
  o, ErrO, oH = forwardPass(X[:-1], X[-1]);
  oGuesses.append(o);
  errSteps.append(ErrO);
  #print('o ', o, ' y ', X[-1]);
  print(ErrO);
  #Wpaspast = np.array(Wpast);
  #Wpast = np.array(W);
  Wfut = np.zeros((nI+nH+nO,nI+nH+nO), dtype='float');
  #print(Wpast);
  #print(W);
  #print(Wfut);
  #print(oH);
  backwardPass(X[:-1], ErrO, o, oH);
  #print(Wfut);
  Wpast = np.array(W);
  W = np.array(Wfut);
  counter+=1;
  #input()

print('Number of iterations ' + str(counter));
fig, ax = plt.subplots((2));
yConstant = [X[-1] for y in range(len(oGuesses))];
constant0 = [0 for y in range(len(oGuesses))];
my_plotter(ax[0], range(len(oGuesses)), oGuesses, {'color':'k'});
my_plotter(ax[0], range(len(oGuesses)), yConstant, {'color':'b', }, title='Prediction vs True value', xlabel='Steps', ylabel='Val.');
my_plotter(ax[1], range(len(oGuesses)), errSteps, {'color':'b'});
my_plotter(ax[1], range(len(oGuesses)), errSteps, {'color':'k'}, title='Error calculation', xlabel='Steps', ylabel='Err.');
fig.tight_layout()
plt.show();





