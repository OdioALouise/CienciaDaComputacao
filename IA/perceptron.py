import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize

#function to plot scatter graphs
def my_plotter_scatter(ax, data1, data2, param_dict, title=None, xlabel=None, ylabel=None):

  out = ax.scatter(data1, data2, **param_dict);

  if title:
    ax.set_title(title);
  if xlabel:
    ax.set_xlabel(xlabel);
  if ylabel:
    ax.set_ylabel(ylabel);

  return out; 

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


def getTrainingSet(dataSet):
  dic = {
    'StripedV'    : 0,
    'StripedD'    : 1,
    'StripedH'    : 3,
    'SingleColor' : 4,
    'Footballer'  : 0,
    'Rugbier'     : 1,
  };

  auxDS = np.array(dataSet);


  for i in range(len(auxDS)):
    auxDS[i,1] = dic[auxDS[i,1]];
    auxDS[i,3] = dic[auxDS[i,3]];
    
  auxDS = np.array(auxDS[:, 1:], dtype='float');

  NX1 = np.sum(auxDS[:, 0]);
  NX2 = np.sum(auxDS[:, 1]); 

  
  auxDS[:, 0] /= NX1;
  auxDS[:, 1] /= NX2;
  
  
  trainingSet = []; #tS[0] bias, tS[1]=T-Shirt, tS[2]=lbs, tS[3]=y
  testSet = [];
  
  for arr in auxDS[:len(auxDS)-6]:
    trainingSet.append([1, arr[0], arr[1], arr[2]]);

  for arr in auxDS[len(auxDS)-6:]:
    testSet.append([1, arr[0], arr[1], arr[2]]);
  

  return trainingSet, testSet;

def activationFunction(u):
  if u >= 0:
    return 1;
  return 0;

def sumFunction(i, x, W, vW):
  indX = np.where(vW == 1);
  return np.sum(W[indX]*x);
  
def modifyW(x, W, vW, err, alpha=0.7):
  indX = np.where(vW == 1);
  
  for i in range(len(indX[0])):
    if indX[0][i] != 0:
      W[indX[0][i], indX[1][i]] = W[indX[0][i], indX[1][i]] + alpha*x[indX[0][i]]*err; 
      
  
def recall(x, W, vW):
  return activationFunction(sumFunction(3, x, W, vW));


dataSet = [ 
  ['James Smith'      , 'StripedH'   , 242.508, 'Rugbier'],
  ['Michael Smith'    , 'SingleColor', 220.462, 'Rugbier'],
  ['James Jhonson'    , 'StripedH'   , 231.485, 'Rugbier'],
  ['William Miller'   , 'StripedV'   , 154.324, 'Footballer'],
  ['Charles Clarck'   , 'StripedD'   , 187.393, 'Footballer'],
  ['Samuel Taylor'    , 'StripedV'   , 165.0  , 'Footballer'],
  ['Daniel Thompson'  , 'StripedH'   , 218.108, 'Rugbier'],
  ['Bjorn Hopcroft'  , 'SingleColor'   , 218.108, 'Rugbier'],
  ['Dennis Satriano'  , 'StripedH'   , 213.108, 'Rugbier'],    
  ['Enzo Francescolli', 'StripedD'   , 198.416, 'Footballer'],
  ['Jose Luis Chilavert', 'StripedV'   , 209.439, 'Footballer'],
  ['Carlos Tevez', 'StripedV'   , 194.007, 'Footballer'],    
  ];

  
trainingSet, testSet = getTrainingSet(dataSet);


vW = np.array([
    [0,0,0,1],
    [0,0,0,1],
    [0,0,0,1],
    [0,0,0,0],
  ]);

#Training Part
#P1 Randomize weights 
W = np.array([
    [0,0,0,1],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
], dtype='float');

W[0,3] = 0.0   #np.random.uniform(0, 1e-1);
W[1,3] = 0.035 #best option #np.random.uniform(0, 1e-1);
W[2,3] = 0.006 #best option #np.random.uniform(0, 1e-1);

#Initial conditions
print('Initial conditions');
print(W[1,3], W[2,3]);

trainingSet = np.array(trainingSet);

def hiperPlane():
  xAxis = [];
  yAxis = []
  for tS in trainingSet:
    x=tS[:-1];
    y=tS[-1];

    xAxis.append(x[2])
    #yAxis.append( (recall(x, W, vW) - x[2]*W[2,3])/W[1,3] );  
    yAxis.append( (-W[0,3] -x[2]*W[2,3])/W[1,3] );  

  return xAxis, yAxis;


fig, ax = plt.subplots();

args = {'color':''}
counter = 0;
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown'];

for tS in trainingSet:

  x=tS[:-1];
  y=tS[-1];

  err = np.Inf;

  numIt = 0;
  
  while(err > 1e-6):
    numIt += 1;

    #P2 Apply an input feature vector x and calculate u(i)
    u =sumFunction(3, x, W, vW);

    #P3 Apply hard-limited treshold activation
    a = activationFunction(u);
    o = a;

    #P4 Compute the error
    err = y - o;

    #P5 Re-compute W
    modifyW(x, W, vW, err);
  xAxis, yAxis = hiperPlane();  
  #Printable purpose
  xP = [arr[2] for arr in trainingSet];
  yP = [arr[1] for arr in trainingSet];


  args['color'] = colors[counter];
  my_plotter(ax, xAxis, yAxis, args);
  my_plotter_scatter(ax, xP[:3], yP[:3], {'marker':'8', 'color':'darkseagreen'});
  my_plotter_scatter(ax, xP[3:], yP[3:], {'marker':'*', 'color':'yellowgreen'}, title=r'Football/Rubgby Classification', xlabel='lbs', ylabel='T-Shirt');
  counter += 1;

plt.legend(['Hyper1','Hyper2', 'Hyper3', 'Hyper4', 'Hyper5', 'Hyper6'], loc='upper left');  
plt.savefig('innerParts.png')


#Recall part

#Best option


print('Training Set')
print(trainingSet);
print(W)

for tS in trainingSet:
  x=tS[:-1];
  y=tS[-1];
  res = recall(x, W, vW);
  print(y, res);

testSet = np.array(testSet);

print('test set ')
for tS in testSet:
  x=tS[:-1];
  y=tS[-1];
  res = recall(x, W, vW);
  print(y, res);

print('end')




#Printable purpose
x = [arr[2] for arr in trainingSet];
y = [arr[1] for arr in trainingSet];

xAxis, yAxis = hiperPlane();

xTest = [ arr[2] for arr in testSet];
yTest = [ arr[1] for arr in testSet];

fig, ax = plt.subplots();
my_plotter(ax, xAxis, yAxis, {'color':'k'});
my_plotter_scatter(ax, x[:3], y[:3], {'marker':'8', 'color':'darkseagreen'});
my_plotter_scatter(ax, x[3:], y[3:], {'marker':'*', 'color':'yellowgreen'}, title=r'Football/Rubgby Classification', xlabel='lbs', ylabel='T-Shirt');

my_plotter_scatter(ax, xTest[:3], yTest[:3], {'marker':'8', 'color':'orchid'});
my_plotter_scatter(ax, xTest[3:], yTest[3:], {'marker':'*', 'color':'darkturquoise'}, title=r'Football/Rubgby Classification', xlabel='lbs', ylabel='T-Shirt');

plt.legend(['Hyper','Rugbiers', 'Footballers', 'TestR', 'TestF'], loc='upper left');
plt.show()


"""
One of the best results
[[ 0.          0.          0.          0.        ]
 [ 0.          0.          0.          0.05861974]
 [ 0.          0.          0.         -0.0736108 ]
 [ 0.          0.          0.          0.        ]]

"""

