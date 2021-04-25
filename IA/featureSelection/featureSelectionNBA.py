import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import train_test_split
pd.set_option("display.max.columns", None)
pd.set_option("display.precision", 2)

df       = pd.read_csv('nbaallelo.csv');
#filterDF = df.query("2011<year_id<=2015 and team_id=='LAL' and is_playoffs==0");
filterDF = df.query("team_id=='LAL' and is_playoffs==0 and (2011<year_id<=2012 or (year_id==2013 and seasongame <= 35))");
dicOppId = {};
dicGL    = {};
winLose  = [];
oppId    = [];
glInfo   = [];

i = 0;
for opId in filterDF['opp_id'].unique():
  dicOppId.update({opId:i});
  i += 1;

i = 0;
for gl in filterDF['game_location'].unique():
  dicGL.update({gl:i});
  i += 1;

def getPlot(LALdataset):

  fig,ax = plt.subplots();
  arr = [];
  label = [];
  for data in LALdataset.groupby('year_id'):
    print('Anio ', data[0]);
    for i in range(0,80,20):
      if i == 60:
        arr.append(data[1]['WL'][i:i+22].mean());
        label.append(str(data[0]) + '/' + str(i) + '-' + str(i+22));
      else:
        arr.append(data[1]['WL'][i:i+20].mean());
        label.append(str(data[0]) + '/' + str(i) + '-' + str(i+20));
  ax.scatter(label, arr);
  ax.set_title('Average of matches won by 20 games/season');
  ax.set_xlabel('N$^\circ$ of game');
  ax.set_ylabel('Average of matches won');
  plt.show();


def getEloDifWInLoose(LALdataset):
  fig, ax = plt.subplots();
  x_axis = np.arange(-500, 500, 0.001)
  for data in LALdataset.groupby('WL'):
    label = 'Win';
    if data[0] == 0:
      label = 'Lose';
    ax.plot(x_axis, norm.pdf(x_axis, data[1]['elo_diff'].mean(), data[1]['elo_diff'].std()), label=label);
    print(label, data[1]['elo_diff'].mean());
  ax.set_label(['Lose', 'Win'])
  ax.set_xlabel('Real values');
  ax.set_ylabel('$\mathcal{N}(\mu,\sigma)$');
  ax.set_title('Normal Distribution elo diff between teams');
  plt.legend();
  plt.show();


def getWinsByNumberSeasonGame(LALdataset):
  fig, ax = plt.subplots();
  arrRes = [];
  arrInd = [];
  counter = 0;
  for data in LALdataset.groupby(['seasongame']):
    arrRes.append(data[1]['WL'].mean());
    arrInd.append(counter);  
    counter += 1;
  plt.plot(arrInd, arrRes);
  ax.set_title('Mean of win games by season game');
  ax.set_xlabel('N$^\circ$ games');
  ax.set_ylabel('Mean');
  plt.show();


def getSeasonGameInfoTeamAway(LALdataset):
  meanArray = [];
  idArray = [];
  abbTeamName = [];
  for data in LALdataset.groupby(['opp_id', 'game_location']):
    if list(dicGL.keys())[list(dicGL.values()).index(data[0][1])] == 'A':
      idArray.append(data[0][0]);
      meanArray.append(data[1]['WL'].mean());
      abbTeamName.append(list(dicOppId.keys())[list(dicOppId.values()).index(data[0][0])]);

  d = {
    'idTeam': idArray,
    'mean' : meanArray,
    'labelTeam' : abbTeamName,
  }
  oppDF = pd.DataFrame(data=d);

  plt.figure(figsize=(8,5))
  sns.scatterplot(data=oppDF,x='idTeam',y='mean');

  for i in range(oppDF.shape[0]):
    plt.text(x=oppDF['idTeam'][i]+0.05,y=oppDF['mean'][i]+0.05,s=oppDF['labelTeam'][i], fontdict=dict(color='red',size=10),bbox=dict(facecolor='yellow',alpha=0.5));

  plt.xlim(oppDF['idTeam'].min()-1,oppDF['idTeam'].max()+1);
  plt.ylim(oppDF['mean'].min(),oppDF['mean'].max()+0.2);

  plt.title("Mean value of Win in Season games Away");
  plt.xlabel('id Teams');
  plt.ylabel('Mean of wins');
  plt.show()  


def getSeasonGameInfoTeamHome(LALdataset):
  meanArray = [];
  idArray = [];
  abbTeamName = [];
  for data in LALdataset.groupby(['opp_id', 'game_location']):
    if list(dicGL.keys())[list(dicGL.values()).index(data[0][1])] == 'H':
      idArray.append(data[0][0]);
      meanArray.append(data[1]['WL'].mean());
      abbTeamName.append(list(dicOppId.keys())[list(dicOppId.values()).index(data[0][0])]);

  d = {
    'idTeam': idArray,
    'mean' : meanArray,
    'labelTeam' : abbTeamName,
  }
  oppDF = pd.DataFrame(data=d);

  plt.figure(figsize=(8,5))
  sns.scatterplot(data=oppDF,x='idTeam',y='mean');

  for i in range(oppDF.shape[0]):
    plt.text(x=oppDF['idTeam'][i]+0.05,y=oppDF['mean'][i]+0.05,s=oppDF['labelTeam'][i], fontdict=dict(color='red',size=10),bbox=dict(facecolor='yellow',alpha=0.5));

  plt.xlim(oppDF['idTeam'].min()-1,oppDF['idTeam'].max()+1);
  plt.ylim(oppDF['mean'].min(),oppDF['mean'].max()+0.2);

  plt.title("Mean value of Win in Season games at Home");
  plt.xlabel('id Teams');
  plt.ylabel('Mean of wins');
  plt.show()  


def getSeasonGameInfo(LALdataset):
  meanArray = [];
  idArray = [];
  abbTeamName = [];
  for data in LALdataset.groupby('opp_id'):
    idArray.append(data[0]);
    meanArray.append(data[1]['WL'].mean());
    abbTeamName.append(list(dicOppId.keys())[list(dicOppId.values()).index(data[0])]);

  d = {
    'idTeam': idArray,
    'mean' : meanArray,
    'labelTeam' : abbTeamName,
  }
  oppDF = pd.DataFrame(data=d);

  plt.figure(figsize=(8,5))
  sns.scatterplot(data=oppDF,x='idTeam',y='mean');

  for i in range(oppDF.shape[0]):
    plt.text(x=oppDF['idTeam'][i]+0.05,y=oppDF['mean'][i]+0.05,s=oppDF['labelTeam'][i], fontdict=dict(color='red',size=10),bbox=dict(facecolor='yellow',alpha=0.5));

  plt.xlim(oppDF['idTeam'].min()-1,oppDF['idTeam'].max()+1);
  plt.ylim(oppDF['mean'].min(),oppDF['mean'].max()+0.2);

  plt.title("Mean value of Win in Season games");
  plt.xlabel('id Teams');
  plt.ylabel('Mean of wins');
  plt.show()  


def getHomeAwayInfo(LALdataset):
  fig, ax = plt.subplots();
  x_axis = np.arange(-5, 5, 0.001)
  for data in LALdataset.groupby(['game_location']):
    label='Home'
    if data[0] == 0:
      label='Away'
    ax.plot(x_axis, norm.pdf(x_axis, data[1]['WL'].mean(), data[1]['WL'].std()), label=label);
  ax.set_label(['Home', 'Away'])
  ax.set_xlabel('Real values');
  ax.set_ylabel('$\mathcal{N}(\mu,\sigma)$');
  ax.set_title('Normal Distribution Win for Game Locations');
  plt.legend();
  plt.show();


def getTrainingData():
  for index, row in filterDF.iterrows():
    if (row['pts'] - row['opp_pts']) > 0:
      winLose.append(1);
    else:
      winLose.append(0);
    oppId.append(dicOppId[row['opp_id']]);
    glInfo.append(dicGL[row['game_location']]);

  filterDF['WL'] = winLose;
  filterDF['opp_id'] = oppId;
  filterDF['game_location'] = glInfo;
  filterDF['elo_diff'] = filterDF['elo_i'] - filterDF['opp_elo_i'];

  LALdataset = filterDF[['year_id','seasongame', 'elo_diff', 'game_location', 'opp_id','WL']];

  #LALdatasetTraining = LALdataset.query('year_id<2013');
  #LALdatasetTest = LALdataset.query('year_id==2013');
  
  #LALdatasetTraining.to_csv('LALdatasetTraining.csv');
  #LALdatasetTest.to_csv('LALdatasetTest.csv');

  #getHomeAwayInfo(LALdataset);
  #getSeasonGameInfo(LALdataset);  
  #getSeasonGameInfoTeamHome(LALdataset);
  #getSeasonGameInfoTeamAway(LALdataset);
  #getEloDifWInLoose(LALdataset);
  

  LALdataset = LALdataset.to_numpy();
  np.save('LALdataset2012extended', LALdataset);

  #X_train, X_test, y_train, y_test = train_test_split(LALdataset[:,:-1], LALdataset[:,-1], stratify=LALdataset[:,-1],random_state=1)
  #print(X_train.shape)
  #print(X_test.shape)
  #print(y_train.shape)
  #print(y_test.shape)

class typeOfFeatureError(Exception):
    def __init__(self, m):
      self.message = m
    def __str__(self):
      return self.message;
  
def class1Feature(X,y,typeOfFeature='Win'):
  if typeOfFeature == 'Win':
    if y == 1:
      return 1;
    else:
      return 0;
  elif typeOfFeature == 'Lose':
    if y == 0:
      return 1;
    else:
      return 0;
  raise typeOfFeatureError('No matching feature');

def class2Feature(X,y, typeOfFeature='LoseNormalAndWin'):
  LoseEloDiffMean = -49.1536717948718;
  WinEloDiffMean  = 41.76400530973452;
  
  absToLose = np.abs(X[0]-LoseEloDiffMean);
  absToWin = np.abs(X[0]-WinEloDiffMean); 
  
  if typeOfFeature == 'LoseNormalAndWin':
    if y and absToLose < absToWin:
      return 1;
    else:
      return 0;
  elif typeOfFeature == 'LoseNormalAndLose':
    if not y and absToLose < absToWin:
      return 1;
    else:
      return 0;
  elif typeOfFeature == 'WinNormalAndWin':
    if y and absToWin < absToLose:
      return 1;
    else: 
      return 0;
  elif typeOfFeature == 'WinNormalAndLose':
    if not y and absToWin < absToLose:
      return 1;
    else:
      return 0;
  elif typeOfFeature == 'CloseToLose':
    if absToLose < absToWin:
      return 1;
    else:
      return 0;
  elif typeOfFeature == 'CloseToWin':
    if absToWin < absToLose:
      return 1;
    else:
      return 0;

  raise typeOfFeatureError('No matching feature');  

def class3Feature(X,y,typeOfFeature='H-W'):

  if typeOfFeature == 'H-W':
    if y and X[1]==0:
      return 1;
    else:
      return 0;
  elif typeOfFeature == 'H-L':
    if not y and X[1] == 0:
      return 1;
    else: 
      return 0;
  elif typeOfFeature == 'A-W':
    if y and X[1] == 1:
      return 1;
    else:
      return 0;
  elif typeOfFeature == 'A-L':
    if not y and X[1] == 1:
      return 1;
    else:
      return 0;
  elif typeOfFeature == 'H':
    if X[1] == 0:
      return 1;
    else:
      return 0;
  elif typeOfFeature == 'A':
    if X[1] == 1:
      return 1;
    else:
      return 0;
      
  raise typeOfFeatureError('No matching feature');  

def class4Feature(X,y,teamId=0,result='W'):
  teamIdArray = [i for i in range(32)];
  if teamId not in teamIdArray:
    raise typeOfFeatureError('Team ID {} out of range, correct range [0-31]'.format(teamId));
  elif X[2] != teamId:
    return 0;
  elif result == 'W' and y:
    return 1;
  elif result == 'W' and not y:
    return 0;
  elif result == 'L' and not y:
    return 1;
  elif result == 'L' and y:
    return 0;
  elif result == '-':
    return 1;
  
  raise typeOfFeatureError('No matching feature {}'.format(result));  
  
def class5Feature(X,y,typeOfFeature='H-W', teamId=0, result='W'):
  return class3Feature(X,y,typeOfFeature=typeOfFeature)*class4Feature(X,y,teamId=teamId,result=result);

def class6Feature(X,y,typeOfFeature='HW'):

  if typeOfFeature == 'HW':
    if y and X[1] == 0 and X[2] in [1,3,20,22,24,26,27,7,9,17,18,10,6,4,12]:
      return 1;
    else:
      return 0;
  elif typeOfFeature == 'HL':
    if not y and X[1] == 0 and X[2] in [15,19,25,28,0,13]:
      return 1;
    else:
      return 0;
  elif typeOfFeature == 'AW':
    if y and X[1] == 1 and X[2] in [18,26,29,12,17,20,19,21,24]:
      return 1;
    else:
      return 0;
  elif typeOfFeature == 'AL':
    if not y and X[1] == 1 and X[2] in [0,3,9,13,16,22,23,30,11,4,5,1,14,25]:
      return 1;
    else:
      return 0;

  raise typeOfFeatureError('No matching feature {}'.format(typeOfFeature));

#DiffElo and Location game
def class7Feature(X,y,typeOfFeatureClass2='LoseNormalAndWin', typeOfFeatureClass3='H-W'):
  return class2Feature(X,y,typeOfFeature=typeOfFeatureClass2)*class3Feature(X,y,typeOfFeature=typeOfFeatureClass3);
#DiffElo and oppID
def class8Feature(X,y,typeOfFeatureClass2='LoseNormalAndWin', teamId=0, result='W'):
  return class2Feature(X,y,typeOfFeature=typeOfFeatureClass2)*class4Feature(X,y,teamId=teamId,result=result);
#DiffElo and Class6
def class9Feature(X,y,typeOfFeatureClass2='LoseNormalAndWin', typeOfFeatureClass6='HW'):
  return class2Feature(X,y,typeOfFeature=typeOfFeatureClass2)*class6Feature(X,y,typeOfFeature=typeOfFeatureClass6);


Features = [
  lambda X,y : class1Feature(X,y,typeOfFeature='Win'),
  lambda X,y : class1Feature(X,y,typeOfFeature='Lose'),  
  lambda X,y : class2Feature(X,y,typeOfFeature='LoseNormalAndWin'),
  lambda X,y : class2Feature(X,y,typeOfFeature='LoseNormalAndLose'),
  lambda X,y : class2Feature(X,y,typeOfFeature='WinNormalAndWin'),
  lambda X,y : class2Feature(X,y,typeOfFeature='WinNormalAndLose'),
  lambda X,y : class2Feature(X,y,typeOfFeature='CloseToWin'),
  lambda X,y : class2Feature(X,y,typeOfFeature='CloseToLose'),
  lambda X,y : class3Feature(X,y,typeOfFeature='H-W'),
  lambda X,y : class3Feature(X,y,typeOfFeature='H-L'),
  lambda X,y : class3Feature(X,y,typeOfFeature='A-W'),
  lambda X,y : class3Feature(X,y,typeOfFeature='A-L'),
  lambda X,y : class3Feature(X,y,typeOfFeature='H'),
  lambda X,y : class3Feature(X,y,typeOfFeature='A'),  
  lambda X,y : class4Feature(X,y,teamId=0,result='W'),
  lambda X,y : class4Feature(X,y,teamId=0,result='L'),
  lambda X,y : class4Feature(X,y,teamId=1,result='W'),
  lambda X,y : class4Feature(X,y,teamId=1,result='L'),
  lambda X,y : class4Feature(X,y,teamId=2,result='W'),
  lambda X,y : class4Feature(X,y,teamId=2,result='L'),
  lambda X,y : class4Feature(X,y,teamId=3,result='W'),
  lambda X,y : class4Feature(X,y,teamId=3,result='L'),
  lambda X,y : class4Feature(X,y,teamId=4,result='W'),
  lambda X,y : class4Feature(X,y,teamId=4,result='L'),
  lambda X,y : class4Feature(X,y,teamId=5,result='W'),
  lambda X,y : class4Feature(X,y,teamId=5,result='L'),
  lambda X,y : class4Feature(X,y,teamId=6,result='W'),
  lambda X,y : class4Feature(X,y,teamId=6,result='L'),
  lambda X,y : class4Feature(X,y,teamId=7,result='W'),
  lambda X,y : class4Feature(X,y,teamId=7,result='L'),
  lambda X,y : class4Feature(X,y,teamId=8,result='W'),
  lambda X,y : class4Feature(X,y,teamId=8,result='L'),
  lambda X,y : class4Feature(X,y,teamId=9,result='W'),
  lambda X,y : class4Feature(X,y,teamId=9,result='L'),
  lambda X,y : class4Feature(X,y,teamId=10,result='W'),
  lambda X,y : class4Feature(X,y,teamId=10,result='L'),
  lambda X,y : class4Feature(X,y,teamId=11,result='W'),
  lambda X,y : class4Feature(X,y,teamId=11,result='L'),
  lambda X,y : class4Feature(X,y,teamId=12,result='W'),
  lambda X,y : class4Feature(X,y,teamId=12,result='L'),
  lambda X,y : class4Feature(X,y,teamId=13,result='W'),
  lambda X,y : class4Feature(X,y,teamId=13,result='L'),
  lambda X,y : class4Feature(X,y,teamId=14,result='W'),
  lambda X,y : class4Feature(X,y,teamId=14,result='L'),
  lambda X,y : class4Feature(X,y,teamId=15,result='W'),
  lambda X,y : class4Feature(X,y,teamId=15,result='L'),
  lambda X,y : class4Feature(X,y,teamId=16,result='W'),
  lambda X,y : class4Feature(X,y,teamId=16,result='L'),
  lambda X,y : class4Feature(X,y,teamId=17,result='W'),
  lambda X,y : class4Feature(X,y,teamId=17,result='L'),
  lambda X,y : class4Feature(X,y,teamId=18,result='W'),
  lambda X,y : class4Feature(X,y,teamId=18,result='L'),
  lambda X,y : class4Feature(X,y,teamId=19,result='W'),
  lambda X,y : class4Feature(X,y,teamId=19,result='L'),
  lambda X,y : class4Feature(X,y,teamId=20,result='W'),
  lambda X,y : class4Feature(X,y,teamId=20,result='L'),
  lambda X,y : class4Feature(X,y,teamId=21,result='W'),
  lambda X,y : class4Feature(X,y,teamId=21,result='L'),
  lambda X,y : class4Feature(X,y,teamId=22,result='W'),
  lambda X,y : class4Feature(X,y,teamId=22,result='L'),
  lambda X,y : class4Feature(X,y,teamId=23,result='W'),
  lambda X,y : class4Feature(X,y,teamId=23,result='L'),
  lambda X,y : class4Feature(X,y,teamId=24,result='W'),
  lambda X,y : class4Feature(X,y,teamId=24,result='L'),
  lambda X,y : class4Feature(X,y,teamId=25,result='W'),
  lambda X,y : class4Feature(X,y,teamId=25,result='L'),
  lambda X,y : class4Feature(X,y,teamId=26,result='W'),
  lambda X,y : class4Feature(X,y,teamId=26,result='L'),
  lambda X,y : class4Feature(X,y,teamId=27,result='W'),
  lambda X,y : class4Feature(X,y,teamId=27,result='L'),
  lambda X,y : class4Feature(X,y,teamId=28,result='W'),
  lambda X,y : class4Feature(X,y,teamId=28,result='L'),
  lambda X,y : class4Feature(X,y,teamId=29,result='W'),
  lambda X,y : class4Feature(X,y,teamId=29,result='L'),
  lambda X,y : class4Feature(X,y,teamId=30,result='W'),
  lambda X,y : class4Feature(X,y,teamId=30,result='L'),
  lambda X,y : class4Feature(X,y,teamId=31,result='W'),
  lambda X,y : class4Feature(X,y,teamId=31,result='L'),
  lambda X,y : class4Feature(X,y,teamId=0,result='-'),
  lambda X,y : class4Feature(X,y,teamId=1,result='-'),
  lambda X,y : class4Feature(X,y,teamId=2,result='-'),
  lambda X,y : class4Feature(X,y,teamId=3,result='-'),
  lambda X,y : class4Feature(X,y,teamId=4,result='-'),
  lambda X,y : class4Feature(X,y,teamId=5,result='-'),
  lambda X,y : class4Feature(X,y,teamId=6,result='-'),
  lambda X,y : class4Feature(X,y,teamId=7,result='-'),
  lambda X,y : class4Feature(X,y,teamId=8,result='-'),
  lambda X,y : class4Feature(X,y,teamId=9,result='-'),
  lambda X,y : class4Feature(X,y,teamId=10,result='-'),
  lambda X,y : class4Feature(X,y,teamId=11,result='-'),
  lambda X,y : class4Feature(X,y,teamId=12,result='-'),
  lambda X,y : class4Feature(X,y,teamId=13,result='-'),
  lambda X,y : class4Feature(X,y,teamId=14,result='-'),
  lambda X,y : class4Feature(X,y,teamId=15,result='-'),
  lambda X,y : class4Feature(X,y,teamId=16,result='-'),
  lambda X,y : class4Feature(X,y,teamId=17,result='-'),
  lambda X,y : class4Feature(X,y,teamId=18,result='-'),
  lambda X,y : class4Feature(X,y,teamId=19,result='-'),
  lambda X,y : class4Feature(X,y,teamId=20,result='-'),
  lambda X,y : class4Feature(X,y,teamId=21,result='-'),
  lambda X,y : class4Feature(X,y,teamId=22,result='-'),
  lambda X,y : class4Feature(X,y,teamId=23,result='-'),
  lambda X,y : class4Feature(X,y,teamId=24,result='-'),
  lambda X,y : class4Feature(X,y,teamId=25,result='-'),
  lambda X,y : class4Feature(X,y,teamId=26,result='-'),
  lambda X,y : class4Feature(X,y,teamId=27,result='-'),
  lambda X,y : class4Feature(X,y,teamId=28,result='-'),
  lambda X,y : class4Feature(X,y,teamId=29,result='-'),
  lambda X,y : class4Feature(X,y,teamId=30,result='-'),
  lambda X,y : class4Feature(X,y,teamId=31,result='-'),
  lambda X,y : class6Feature(X,y,typeOfFeature='HW'),
  lambda X,y : class6Feature(X,y,typeOfFeature='HL'),
  lambda X,y : class6Feature(X,y,typeOfFeature='AW'),
  lambda X,y : class6Feature(X,y,typeOfFeature='AL'),
  lambda X,y : class9Feature(X,y,typeOfFeatureClass2='LoseNormalAndWin', typeOfFeatureClass6='HW'),
  lambda X,y : class9Feature(X,y,typeOfFeatureClass2='LoseNormalAndLose', typeOfFeatureClass6='HW'), 
  lambda X,y : class9Feature(X,y,typeOfFeatureClass2='WinNormalAndWin', typeOfFeatureClass6='HW'),
  lambda X,y : class9Feature(X,y,typeOfFeatureClass2='WinNormalAndLose', typeOfFeatureClass6='HW'),
  lambda X,y : class9Feature(X,y,typeOfFeatureClass2='LoseNormalAndWin', typeOfFeatureClass6='HL'),
  lambda X,y : class9Feature(X,y,typeOfFeatureClass2='LoseNormalAndLose', typeOfFeatureClass6='HL'), 
  lambda X,y : class9Feature(X,y,typeOfFeatureClass2='WinNormalAndWin', typeOfFeatureClass6='HL'),
  lambda X,y : class9Feature(X,y,typeOfFeatureClass2='WinNormalAndLose', typeOfFeatureClass6='HL'),
  lambda X,y : class9Feature(X,y,typeOfFeatureClass2='LoseNormalAndWin', typeOfFeatureClass6='AW'),
  lambda X,y : class9Feature(X,y,typeOfFeatureClass2='LoseNormalAndLose', typeOfFeatureClass6='AW'), 
  lambda X,y : class9Feature(X,y,typeOfFeatureClass2='WinNormalAndWin', typeOfFeatureClass6='AW'),
  lambda X,y : class9Feature(X,y,typeOfFeatureClass2='WinNormalAndLose', typeOfFeatureClass6='AW'),
  lambda X,y : class9Feature(X,y,typeOfFeatureClass2='LoseNormalAndWin', typeOfFeatureClass6='AL'),
  lambda X,y : class9Feature(X,y,typeOfFeatureClass2='LoseNormalAndLose', typeOfFeatureClass6='AL'), 
  lambda X,y : class9Feature(X,y,typeOfFeatureClass2='WinNormalAndWin', typeOfFeatureClass6='AL'),
  lambda X,y : class9Feature(X,y,typeOfFeatureClass2='WinNormalAndLose', typeOfFeatureClass6='AL'),  
  ]

#Input: dataSet
#Output: Empirical Probability empXY
def empiricalPXY(dataSet):
  nR, _ = dataSet.shape;
  empXY = np.zeros(nR); #Initialize the empirical probabilities to 0

  for i in range(nR): #For each pair X,y
    count = 0;
    for j in range(nR): #Count their match in the dataset
      if (dataSet[i] == dataSet[j]).all():
        count+=1;
    count /= nR; #Divide by N
    empXY[i] = count; #Update the i-th empirical probability

  return empXY; 

#Input: X,y independent variable X/result y
#Output: Number of features f that match with X,y
def fNumeral(X,y,F):
  count = 0; #Initialize counting matches
  for f in F: #For every featur test if it matches with X,y
    count += f(X,y);
  return count; #Return count of matches

def g(Delta, f, F, dataSet, testMode=False, testMarker=[]):
  nR, _ = dataSet.shape;
  
  #For each x,y
  term1 = 0;
  term2 = 0;
  for i in range(nR):
    if not testMode:
      if globalMarker[i]:
        term1 += dataSet[i,-2]*dataSet[i,-1]*f(dataSet[i,:-4], dataSet[i,-4])*np.exp(Delta*fNumeral(dataSet[i,:-4], dataSet[i,-4], F));
        term2 += dataSet[i,-3]*f(dataSet[i,:-4], dataSet[i,-4]);
    elif testMode and len(testMarker):
      if testMarker[i]:
        term1 += dataSet[i,-2]*dataSet[i,-1]*f(dataSet[i,:-4], dataSet[i,-4])*np.exp(Delta*fNumeral(dataSet[i,:-4], dataSet[i,-4], F));
        term2 += dataSet[i,-3]*f(dataSet[i,:-4], dataSet[i,-4]);
   
  return term1 - term2;

def gPrime(Delta, f, F, dataSet, testMode=False, testMarker=[]):
  nR, _ = dataSet.shape;
  computation = 0;
    
  #For each x,y
  for i in range(nR):
    if not testMode:
      if globalMarker[i]:
        computation += dataSet[i,-2]*dataSet[i,-1]*f(dataSet[i,:-4], dataSet[i,-4])*fNumeral(dataSet[i,:-4], dataSet[i,-4], F)*np.exp(Delta*fNumeral(dataSet[i,:-4], dataSet[i,-4], F));
    elif testMode and len(testMarker):
      if testMarker[i]:
        computation += dataSet[i,-2]*dataSet[i,-1]*f(dataSet[i,:-4], dataSet[i,-4])*fNumeral(dataSet[i,:-4], dataSet[i,-4], F)*np.exp(Delta*fNumeral(dataSet[i,:-4], dataSet[i,-4], F));
  return computation;


#Input iterative value Delta_i, feature f_i, set of features F, dataSet
#Output: The convergent value of Delta_i or np.nan
def newtonMethod(Delta, f, F, dataSet, testMode=False, testMarker=[]):
  maxSteps = 350; #Max number of iterations accepted
  counterSteps = 0;

  while(counterSteps < maxSteps): #Steps condition
    counterSteps += 1;
    if not testMode: 
      Delta = Delta - g(Delta,f, F, dataSet)/gPrime(Delta, f, F, dataSet); #Iteration with Newton's Method formula
    else: #Test mode block
      Delta = Delta - g(Delta,f, F, dataSet, testMode=True, testMarker=testMarker)/gPrime(Delta, f, F, dataSet, testMode=True, testMarker=testMarker);    
      if np.abs(g(Delta,f,F,dataSet, testMode=True, testMarker=testMarker))<1E-3:
        return Delta;
      else:
        continue;
    if np.isnan(Delta):
      break;
    elif np.abs(g(Delta,f,F,dataSet))<1E-3: #Stop condition, convergence with g(Delta)-0 < 0.001
      break;
  return Delta;

#Input F set of features, dataSet with empirical probability information, 
#actual set of optimal lambdas
#Output: optimal lambdas, and optimal model p(y|X)
def iterativeScaling(F, dataSet,lambdas=[]):
  nF = len(F); #Obtain size of the dataSet and Features
  nR, _ = dataSet.shape;

  #Step 1 - Initialize lambdas if there aren't
  if len(lambdas) == 0:
    lambdas = np.zeros(nF);

  #Step 2 - For each feature
  for i in range(nF):
    #Step 2a - Compute deltaLambda_i
    deltaLambda = newtonMethod(lambdas[i], F[i], F, dataSet);
    #Step 2b - Update lambda
    lambdas[i] += deltaLambda;
    if np.isnan(deltaLambda): 
      break;
  #Compute optimalModel
  model = np.zeros(nR); #Initialize array of p(y|X) to 0
  
  if (np.isnan(lambdas)).any(): #Check that all the lambdas have valid values,
  #if not return this
    lambdas[:] = -np.inf;
    return lambdas, model;
  
  for i in range(nR): #For each X,y calculate p(y|X)
    exponentialSum = 0;
    for j in range(nF): #Sum for each lambda_j * feature_j
      exponentialSum = lambdas[j]*F[j](dataSet[i,:-4], dataSet[i,-4]);
    model[i] = np.exp(exponentialSum)/(1+np.exp(exponentialSum));
  return lambdas, model;

#Input: candidate features F, set of optimal lambdas
#dataSet with empirical probability information
#Output: logLikeliHood of the model
def logLikeliHood(F, lambdas, dataSet):
  firstTerm = 0; #First term of the Psi(lambda) equation
  secondTerm = 0; #Second term
  nR, _ = dataSet.shape; #Size of the dataSet

  for i in range(nR): #For row in the dataset
    auxFirstTerm = 0;
    for j in range(len(F)): #Compute the firstTerm
      auxFirstTerm += lambdas[j]*F[j](dataSet[i,:-4], dataSet[i,-4]);
    firstTerm += -dataSet[i,-2]*np.log(1 + np.exp(auxFirstTerm));

  for i in range(len(F)): #For each feature in F
    auxSecondTerm = 0;
    for j in range(nR): #Compute the second term
      auxSecondTerm += dataSet[j,-3]*F[i](dataSet[j,:-4], dataSet[j,-4]);
    secondTerm += lambdas[i]*auxSecondTerm;
    
  return firstTerm + secondTerm;

  
def featureSelection(F, dataSet, testSet):
  #Step 1 - Start With S = empty set
  S = [
#  lambda X,y : class1Feature(X,y,typeOfFeature='Win'),
#  lambda X,y : class1Feature(X,y,typeOfFeature='Lose'),
#  lambda X,y : class8Feature(X,y,typeOfFeatureClass2='WinNormalAndWin', teamId=1, result='W'),    
  ];
  lambdaVector = []; #[10.31460693,  3.40358738,  1.21308964]; #Initialize optimal lambdas for training set

  nF,_ = dataSet.shape;

  dataSet = np.append(dataSet, np.zeros((nF,1)), axis=1); #Update cond. probabilities of the Training Set
  pS = 1/nF;
  dataSet[:,-1] = pS;
  nFTest, _ = testSet.shape;
  testSet = np.append(testSet, np.zeros((nFTest,1)), axis=1); #Update cond. probabilities of the Test Set
  pSTest = 1/nFTest;
  testSet[:,-1] = pSTest;

  pastLogLH = 0; #Past logLikeliHood

  fig, ax = plt.subplots();
  xAxis = [];
  yAxis = [];
  yAxisTest = [];

  stepCounter = 0; #Number of features added


  while(stepCounter < 10):
    logLHArray = np.zeros(len(F));#Array of logLikeliHoods
    
    #Step 2 - For each feature
    counterProcess = 0; #Number of features processed in Foreach feature
    for f in F:
      #Step 2 a - Compute SUf
      SUf = np.append(S,f);
      lambdasF, _ = iterativeScaling(SUf, dataSet, np.append(lambdaVector, [0]));
      if (np.isnan(lambdasF)).any() or np.isinf(lambdasF).any():
        logLHArray[counterProcess] = -np.inf;
        counterProcess += 1;
        continue;
      logLHArray[counterProcess] = logLikeliHood(SUf, lambdasF, dataSet)-pastLogLH;
      if np.isnan(logLHArray[counterProcess]):
        logLHArray[counterProcess] = -np.inf;
      print('Adding feature {} Counter process {} {:.4f}'.format(stepCounter, counterProcess, logLHArray[counterProcess]), end='\r');
      #print('');
      counterProcess += 1;
  
    maxNum = np.max(logLHArray);
    nextF = np.random.choice(np.where(logLHArray == maxNum)[0]);

    xAxis.append(stepCounter);
    yAxis.append(logLHArray[nextF]+pastLogLH);

    #Step 5 adjoint the chosen feature to the set of features S
    S = np.append(S, [F[nextF]], axis=0); #Pattern of equiprobability
    F = np.delete(F, nextF, axis=0);
    pastLogLH = np.array(logLHArray)[nextF] + pastLogLH;

    #Step 6 compute pS with the chosen features
    lambdaVector, dataSet[:,-1] = iterativeScaling(S, dataSet, np.append(lambdaVector, [0]));


    #Compute the logLikeLihood for the test set
    _, testSet[:,-1] = iterativeScaling(S, testSet, lambdaVector);
    logLHTest = logLikeliHood(S, lambdaVector, testSet);
    if np.isnan(logLHTest):
      logLHTest = -np.inf;
    yAxisTest.append(logLHTest);

    print('\n NEXT FEATURE {}'.format(nextF), ' - lambdas ', lambdaVector, ' logTraining ', pastLogLH, ' logTest ', logLHTest,' \n');
    ax.clear();
    ax.scatter(xAxis, yAxis, label='Train.');
    ax.scatter(xAxis, yAxisTest, label='Test');
    ax.set_xlabel('N$^\circ$ features');
    ax.set_ylabel('logLH');
    ax.set_title('Loglikelihood of the model vs Number of restrictions');
    ax.legend();
    plt.savefig('NBALLH.png');

    stepCounter += 1;

  
#Mark with 1 if the row of the dataset is used for computation in the featureSelection, Newton's Method and Iterative Scaling algorithm
globalMarker = [];

if __name__ == '__main__':
  #Obtain the csv version of the Training and the Test csv
  #getTrainingData();

  #Get the statistics of LAL from 2012 to 2014, regular season
  #LALdataset = pd.read_csv('LALdatasetTraining.csv');
  #LALdataset = pd.read_csv('LALdatasetTraining.csv');

  #getPlot(LALdataset);
  #getHomeAwayInfo(LALdataset);
  #getSeasonGameInfo(LALdataset);  
  #getSeasonGameInfoTeamHome(LALdataset);
  #getSeasonGameInfoTeamAway(LALdataset);
  #getEloDifWInLoose(LALdataset);
  #print(dicOppId);
  # DATASET COLUMNS [['elo_diff', 'game_location', 'opp_id','WL']];
  # EXTENDED DATASET COLUMNS [['elo_diff', 'game_location', 'opp_id','WL', EMPPXY, EMPX, CONDY|X]];

  #LALdataset = np.load('LALdataset.npy');   
  #LALdataset = np.load('LALdataset2012.npy');
  LALdataset = np.load('LALdataset2012extended.npy');
  indxTrain2012 = np.where(LALdataset[:,0] == 2012);
  indxTest2013 = np.where(LALdataset[:,0] == 2013);
  LALdataset = LALdataset[:,2:];
  LALTrainingSet = LALdataset[indxTrain2012];
  LALTestSet = LALdataset[indxTest2013];
  
  #Configure the Training Set
  empXY = empiricalPXY(LALTrainingSet);
  empX = empiricalPXY(LALTrainingSet[:,:-1]);
  extendedLALTrainingSet = np.append(LALTrainingSet, np.append(empXY.reshape(-1,1), empX.reshape(-1,1), axis=1), axis=1);

  #Configure the Test Set
  empXYTest = empiricalPXY(LALTestSet);
  empXTest = empiricalPXY(LALTestSet[:,:-1]);
  extendedLALTestSet = np.append(LALTestSet, np.append(empXYTest.reshape(-1,1), empXTest.reshape(-1,1), axis=1), axis=1);


  for i in range(len(extendedLALTrainingSet)):
    mark = 1
    for j in range(i-1,-1,-1):
      if (extendedLALTrainingSet[i,:-3] == extendedLALTrainingSet[j,:-3]).all():
        mark = 0;
        break;
    globalMarker.append(mark);


  featureSelection(Features, extendedLALTrainingSet, extendedLALTestSet);
