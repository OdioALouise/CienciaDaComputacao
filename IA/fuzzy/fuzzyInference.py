#!/usr/bin/python3
import numpy as np;
import pandas as pd;
import seaborn as sns;
import matplotlib.pyplot as plt;

sns.color_palette("husl", 8);

#u - value of the activation function fU
#v - value of the activation function fV

def Rm(u,v):
  return max(min(u, v),(1-u));

def Rs(u,v):
  if u <= v:
    return 1;
  return 0;

def Rb(u,v):
  return max(1-u, v);

#Very Fuzzy Set
def verySet(u):
  return u**2;

#Identity Fuzzy Set
def idSet(u):
  return u;

#Max-min composition function to obtain V from U'*R(U,V)
#v fixed-value
#uList elements of the universe U to vary in the formula
#R relation involved in the composition
#modF, modifier function to the knowledge uA, the modifier may be
#not uA, uA**2, sqrt(uA), etc,...
def max_min(v, uList, R, modF=idSet):

  nValues = len(uList);
  calcVal = [];

  for i in range(nValues):
    calcVal.append(min(modF(uList[i]),R(uList[i], v))); #Chose the mins

  return max(calcVal), np.array(calcVal); #Chose the max


def drawRelation(R):
  uU = np.linspace(0,1,11); #activation function values for the set U
  uV = np.linspace(0,1,11);#activation function values for the set V
  RVList = []; #values of the R list
  UList = []; #uU[i] value for the RVList[i] relation
  VList = []; #uV[i] value for the RVList[i] relation   

  nF = int(len(uV)/3);
  nC = 3;
  if nC:
    nF+=1;
  fig, axs = plt.subplots(nF, nC, sharey=True);
  contSubPlots = 0;

  for i in range(len(uU)):
    subPlotRVList = [];
    for j in range(len(uV)):
      RVList.append(R(uU[i], uV[j]));
      UList.append(uU[i]);
      VList.append(uV[j]);

  for i in range(len(uV)):
    subPlotRVList = [];
    for j in range(len(uU)):
      subPlotRVList.append(R(uU[j], uV[i]));

    axs[int(contSubPlots/3), contSubPlots%3].set_title('R(u,v), v:{:.1f}'.format(uV[i]));
    axs[int(contSubPlots/3), contSubPlots%3].set_xlabel('u');
    if contSubPlots%3 == 0:
      axs[int(contSubPlots/3), contSubPlots%3].set_ylabel('R(u,v)');    
    axs[int(contSubPlots/3), contSubPlots%3].plot(uV,subPlotRVList);
    axs[int(contSubPlots/3), contSubPlots%3].set_ylim(ymax = 1, ymin = 0);
    contSubPlots+=1;

  plt.subplots_adjust(hspace=1.0, wspace=0.4);
  plt.show();
  plt.clf();

  d = {
    'u':UList,
    'v':VList,
    'R':RVList
  };
  
  df = pd.DataFrame(d);



  #for tpl in df.groupby('u'):
  #  df2 = tpl[1].iloc[[-1]].to_numpy();
  #  plt.text(1, df2[0,0], "{:.1f}".format(df2[0,0]), size='medium', color='black', weight='semibold');

  sns.lineplot(data=df, x='u', y='R', hue='v');
  plt.title('R(u,v)');
  plt.show();
  plt.clf();

  #for tpl in df.groupby('v'):
  #  df2 = tpl[1].iloc[[-1]].to_numpy();
  #  plt.text(1, df2[0,1], "{:.1f}".format(df2[0,1]), size='medium', color='black', weight='semibold');

  #sns.lineplot(data=df, x='v', y='R', hue='u');
  #plt.show();


def drawComposition(Comp, R, modF=idSet):
  vList = np.linspace(0,1,11);
  uList = np.linspace(0,1,101);
  nU = len(uList);
  uListCol = [];
  vListCol = [];
  compRes = [];
  valComp = [];
  valReal = [];


  nF = int(len(vList)/3);
  nC = 3;
  if nC:
    nF+=1;
  fig, axs = plt.subplots(nF, nC);
  contSubPlots = 0;
  
  for v in vList:
    val, vectVal = Comp(v, uList, R, modF);
    compRes = np.append(compRes, vectVal);
    print(np.where(vectVal == val))
    uListCol = np.append(uListCol, uList);
    vListCol = np.append(vListCol, [v for _ in range(nU)]);
    valComp.append(val);
    valReal.append(v);
    axs[int(contSubPlots/3), contSubPlots%3].set_xlabel('u');
    if contSubPlots%3 == 0:
      axs[int(contSubPlots/3), contSubPlots%3].set_ylabel('min-max(u,v)');    
    axs[int(contSubPlots/3), contSubPlots%3].set_title('v:{:.1f}'.format(v));
    axs[int(contSubPlots/3), contSubPlots%3].plot(uList,vectVal);
    if len(np.where(vectVal == val)[0]) == 1:
      axs[int(contSubPlots/3), contSubPlots%3].scatter(uList[np.where(vectVal == val)],vectVal[np.where(vectVal == val)], **{'marker':'.', 'color':'k'});    
    else:
      axs[int(contSubPlots/3), contSubPlots%3].plot(uList[np.where(vectVal == val)],vectVal[np.where(vectVal == val)]);    
    axs[int(contSubPlots/3), contSubPlots%3].set_ylim(ymax = 1, ymin = 0);
    axs[int(contSubPlots/3), contSubPlots%3].text(val,vectVal[np.where(vectVal == val)[0][0]],"{:.1f}:{:.2f}".format(v,val));
    contSubPlots+=1;

  plt.subplots_adjust(hspace=1.0, wspace=0.4);
  plt.show();
  plt.clf();
    
  d = {
    'u' : uListCol,
    'v' : vListCol,
    'compF': compRes,
  };
  
  df = pd.DataFrame(d);

  #for tpl in df.groupby('u'):
  #  df2 = tpl[1].iloc[[-1]].to_numpy();
  #  if df2[0,0] in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
  #   plt.text(1, df2[0,0], "{:.1f}".format(df2[0,0]), size='medium', color='black', weight='semibold');

  
  sns.lineplot(data=df, x='u', y='compF', hue='v');
  plt.ylabel('min-max(u,v)');
  plt.show();

  d = {
    'v':valReal,
    "v'":valComp,
  }

  df = pd.DataFrame(d);

  sns.lineplot(data=df,x='v', y="v'");
  plt.plot(np.linspace(0,1,101), np.power(np.linspace(0,1,101),2));
  plt.title("Real v VS. Infered v'");
  plt.show();

def triangularMF(U, a, b):
  res = [];
  for u in U:
    if(np.abs(u-b)<np.abs(b-a)): 
      res.append(1 - np.abs(u-b)/np.abs(b-a));
    else:
      res.append(0);
  return np.array(res);

def drawFuzzyMemb():

  b=0.48;
  d=0.24;
  a = b-d;
  c = b+d;  

  U = np.linspace(0,1,1000);
  mf = lambda U,a,b : triangularMF(U,a,b);
  
  fix, ax = plt.subplots();
  ax.plot(U, mf(U,a,b), label='$\mu_{A}$');
  ax.plot(U, np.power(mf(U,a,b),2), label='$\mu_{A}^2$');
  ax.plot(U, np.sqrt(mf(U,a,b)), label='$\sqrt{\mu_{A}}$');
  ax.plot(U,1 - mf(U,a,b), label='$\lnot \mu_{A}$');
  ax.set_xlabel('U');
  ax.set_title('Unary operations on $\mu_{A}$');
  plt.legend();
  plt.show();

if __name__ == '__main__':
  #drawRelation(Rm);
  #drawRelation(Rs);
  #drawRelation(Rb);
  #drawComposition(max_min, Rm, modF=verySet);
  drawComposition(max_min, Rs, modF=verySet); 
  #drawComposition(max_min, Rb, modF=verySet); 
  #drawFuzzyMemb();
  
  
  
