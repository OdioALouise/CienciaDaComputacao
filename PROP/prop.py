# -*- coding: utf-8 -*-
import re

binConnectives = '↔|→|∨|∧';
unConnective   = '¬';
atom            = 'p\d+';
falsum          = '⊥';

atomicMatch  = '^%s$'%(atom);
falsumMatch  = '^%s$'%(falsum);
binConnMatch = '^\((\(.+?\)|p\d+|⊥)(%s)(\(.+?\)|p\d+|⊥)\)$'%(binConnectives);
unConnMatch  = '^\((%s)(\(.+?\)|p\d+|⊥)\)$'%(unConnective);

class Tree:
  def __init__(self, root, left=None, right=None):
    self.root = root;
    self.left = left;
    self.right = right;

  def printTree(self, tab=''):
    print("%s%s \n"%(tab, self.root));
    if self.left:
      self.left.printTree(tab = tab + '   ');
    if self.right:
      self.right.printTree(tab = tab + '   ');

def parseTree(phi):
  if re.match(atomicMatch, phi) or re.match(falsumMatch, phi):
    return Tree(phi);
  elif re.match(binConnMatch, phi):
    groups = re.match(binConnMatch, phi);
    return Tree(groups.group(2), left = parseTree(groups.group(1)), right = parseTree(groups.group(3)));
  elif re.match(unConnMatch, phi):
    groups = re.match(unConnMatch, phi);
    return Tree(groups.group(1), left = parseTree(groups.group(2)))

def checkProp(phi):
  queue = [];
  queue.insert(0, phi)
  while(queue):
    psi = queue.pop(0);
    if re.match(atomicMatch, psi) or re.match(falsumMatch, psi):
      continue;
    elif re.match(binConnMatch, psi):
      groups = re.match(binConnMatch, psi);
      queue.insert(0, groups.group(1));
      queue.insert(0, groups.group(3));
    elif re.match(unConnMatch, psi):
      groups = re.match(unConnMatch, psi);
      queue.insert(0, groups.group(2));
    else:
      return False;
  return True;

if __name__ == '__main__':
  stringTree1 = '(p0↔(p1∧p0))'; 
  stringTree2 = '(¬((p1∧p0)↔(p1∧p0)))';
  stringTree3 = '((p1∧p0)↔((p1∧p0)→(¬p3)))';
  stringTree4 = '(p0)';
  stringTree5 = '((⊥';
  stringTree6 = '(¬p0)';

  print('First tree');
  if checkProp(stringTree1):
    t = parseTree(stringTree1);
    t.printTree();

  print('Second tree');
  if checkProp(stringTree2):
    t = parseTree(stringTree2);
    t.printTree();

  print('Third tree');
  if checkProp(stringTree3):
    t = parseTree(stringTree3);
    t.printTree();

  print('Fourth tree');
  if checkProp(stringTree4):
    t = parseTree(stringTree4);
    t.printTree();

  print('Fifth tree');
  if checkProp(stringTree5):
    t = parseTree(stringTree5);
    t.printTree();

  print('Sixth tree');
  if checkProp(stringTree6):
    t = parseTree(stringTree6);
    t.printTree();
