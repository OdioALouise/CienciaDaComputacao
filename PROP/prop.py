# -*- coding: utf-8 -*-
import re;
import numpy as  np;
import pandas as pd;
from sklearn.metrics import f1_score

binConnectives = u'↔|→|∨|∧';
unConnective   = u'¬';
atom            = u'p\d+';
falsum          = u'⊥';



def checkProp(phi):
  #phi só tem símbolos do alfabeto de PROP
  if re.search(u'[^↔→∨∧⊥¬p\d\(\)]', phi) != None:
    return False;
  #crear fila de espera
  queue = [];
  queue.insert(0, phi);
  #enquanto a fila de espera tenha textos a ser evaluados fazer
  while(queue):
    #obter texto psi candidato a ser proposição
    psi = queue.pop(0)
    #test1 psi é atomico ou falsum 
    if re.match(u'^(%s|%s)$'%(atom, falsum), psi):
      continue;
     #test2 psi e uma negação
    elif re.match(u'^\((%s)(.+)\)$'%(unConnective), psi):
      groups = re.match(u'^\((%s)(.+)\)$'%(unConnective), psi);
      queue.insert(0, groups.group(2));
    #test3 psi e uma proposição com conetiva central binaria
    elif re.match(u'^\((.+)(%s)(.+)\)$'%(binConnectives), psi):
      psi = psi[1:-1];
      countLB = 0;
      countRB = 0;
      for i,s in enumerate(psi):
        if re.match('%s'%(binConnectives), s):
          if countLB == countRB:
            queue.insert(0,psi[:i]);
            queue.insert(0,psi[i+1:]);
            break;
        elif s == u'(':
          countLB += 1;
        elif s == u')':
          countRB += 1 
      if countLB != countRB:
        return False;
    else:
      return False;
  return True;

if __name__ == '__main__':
  d = {
   'PROPS' : [
      u'Hello world!',
      u'p0 is an atom.',
      u'p0p5p9p9p12',
      u'((((p0↔(¬p13))))))',
      u'(⊥)',
      u'(p0)',
      u'(p0↔⊥)',
      u'(p0↔⊥)∨(p0↔⊥)',
      u'((p0↔⊥)∨(p0↔⊥))',
      u'p0',
      u'(¬p0)',
      u'(p0↔p40)',
      u'(¬(p0↔p40))',
      u'((¬(p0↔p40))∨(p1100∧⊥))',
      u'⊥⊥',
      u'((((p0→p1)→p2)→p0)→p15)',
      u'(¬(¬(¬((((p0→p1)→p2)→p0)→p15))))',
      u'((¬(¬(¬((((p0→p1)→p2)→p0)→p15))))∧(¬(¬(¬((((p0→p1)→p2)→p0)→p15)))))',
      u'(((¬(¬(¬((((p0→p1)→p2)→p0)→p15))))∧(¬(¬(¬((((p0→p1)→p2)→p0)→p15)))))↔⊥)',
      u'((((¬(¬(¬((((p0→p1)→p2)→p0)→p15))))∧(¬(¬(¬((((p0→p1)→p2)→p0)→p15)))))↔⊥)→(((¬(¬(¬((((p0→p1)→p2)→p0)→p15))))∧(¬(¬(¬((((p0→p1)→p2)→p0)→p15)))))↔⊥))',
      u'(((((¬(¬(¬((((p0→p1)→p2)→p0)→p15))))∧(¬(¬(¬((((p0→p1)→p2)→p0)→p15)))))↔⊥)→(((¬(¬(¬((((p0→p1)→p2)→p0)→p15))))∧(¬(¬(¬((((p0→p1)→p2)→p0)→p15)))))↔⊥))∨((((¬(¬(¬((((p0→p1)→p2)→p0)→p15))))∧(¬(¬(¬((((p0→p1)→p2)→p0)→p15)))))↔⊥)→(((¬(¬(¬((((p0→p1)→p2)→p0)→p15))))∧(¬(¬(¬((((p0→p1)→p2)→p0)→p15)))))↔⊥)))',
u'((((¬(¬(¬((((p0→p1)→p2)→p0)→p15))))∧(¬(¬(¬((((p0→p1)→p2)→p0)→p15)))))↔⊥)→(((¬(¬(¬((((p0→p1)→p2)→p0)→p15))))∧(¬(¬(¬((((p0→p1)→p2)→p0)→p15)))))↔⊥))∨((((¬(¬(¬((((p0→p1)→p2)→p0)→p15))))∧(¬(¬(¬((((p0→p1)→p2)→p0)→p15)))))↔⊥)→(((¬(¬(¬((((p0→p1)→p2)→p0)→p15))))∧(¬(¬(¬((((p0→p1)→p2)→p0)→p15)))))↔⊥)))',
      u'(((p0→p1)∧(p2→p3))∨(¬p1∧¬p2))',
      u'(((p0→p1)∧(p1→p2)∨(¬p0∧¬p1))',
      u'((p0→p1)→(p1p1))p0))',
      u'(((p0→p1)∧(p2→p3))∨((¬p1)∧(¬p2)))',
      u'(((())))',
      u'()',
      u'',
      u' ',
    ],
    'goldTag' : [
      False,
      False,
      False,
      False,
      False,
      False,
      True,
      False,
      True,
      True,
      True,
      True,
      True,
      True,
      False,
      True,
      True,
      True,
      True,
      True,
      True,
      False,
      False,
      False,
      False,
      True,
      False,
      False,
      False,
      False,
    ]

  }

  df = pd.DataFrame(d);
  df['isProp'] = df['PROPS'].apply(checkProp);

  print(df);
  print(f1_score(df['goldTag'], df['isProp']))
