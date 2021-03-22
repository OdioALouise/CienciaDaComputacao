#Regular expression library
import re

#Simulation of requests from a User
requests = [
  "Hi",
  "My name is Menelaus",
  "I'm feeling sad",
  "because my brother teases me",
  "because my brother teases me and my girlfriend didn't like it my dog"
  ]

#Regular expressions managed by the bot
patterns = [
  #r"[Hh]ello|Hi!?",
  r"H(ELLO|I)!?",
  r"MY NAME IS ([A-Z]+)",
  r"SAD|ANGRY|ANXIOUS|DESPERATE|HORRIBLE",
  r"HAPPY|WONDERFUL|GREAT|SPECTACULAR|INCREDIBLE",
  r".*\b(MY\s\w+\s|THE\s\w+\s)\b(.*)(ME)(?!.*AND).*",
  r".*\b(MY\s\w+\s|THE\s\w+\s)\b(.*)(ME).*AND.*?\b(MY|THE)\b(.*)(MY.*)"  
]

#Template responses from the bot
responses = [
  r"Bot : <SUBS> WHAT IS YOUR NAME?",
  r"Bot : TELL ME <SUBS>, HOW DO YOU FEEL?",
  r"Bot : DEAR <NAME>, TELL ME MORE ABOUT THAT <SUBS> FEELING PLEASE",
  r"Bot : AND WHY DO YOU THINK THAT <SUBS1> <SUBS2> <SUBS3>?",
  r"Bot : <NAME>, I WANT TO DIG MORE IN THE DETAILS WHY <SUBS1> <SUBS2> <SUBS3> AND <SUBS4> <SUBS5> <SUBS6>?"
]

#Class for save data from the User
class Person():
  def __init__(self, name=''):
    self.name = name
  
  def getName(self):
    return self.name;
    
  def setName(self, name):
    self.name = name;
    
#Pre-processing of the requests, all the letters are converted to UPPERCASE
#and the contraction I'M and N'T are converted to 'I AM' and ' NOT'
def requestPreProc(requests):
  copyReq = requests.copy();
  for i in range(len(copyReq)):
    copyReq[i] = copyReq[i].upper();
    copyReq[i] = re.sub("I'M", "I AM", copyReq[i]);
    copyReq[i] = re.sub("N'T", " NOT", copyReq[i]);
  return copyReq;


person = Person();

#Pre-processing of the requests  
copyReq = requestPreProc(requests);


#Logic
#For every User's request pre-processed 'req' apply in order ever patter 'ptn'
#finally dependig the pattern[i] matched the response given
for req in copyReq:
  print('User: ' + req);
  for i, ptn in enumerate(patterns):
    m = re.search(ptn, req);
    if(m):
      if i == 0:
        print(re.sub('<SUBS>', m[0], responses[0]));
        break;
      if i == 1:
        name = re.findall(ptn, req)[0];
        person.setName(name);        
        print(re.sub('<SUBS>', person.getName(), responses[1]));

        break;
      if i == 2:
        print(re.sub('<NAME>', person.getName(), re.sub('<SUBS>', m[0], responses[2])))
        break;
      if i == 4:
        resList = list(re.findall(ptn, req)[0]);
        for i in range(len(resList)):
          resList[i] = re.sub('MY', 'YOUR', resList[i]);
          resList[i] = re.sub('ME', 'YOU', resList[i]);
        print(re.sub(
              '<SUBS3>', resList[2], re.sub(
                '<SUBS2>', resList[1], re.sub(
                   '<SUBS1>', resList[0], responses[3]
                 )
               )
            ))      
        break;
      if i == 5:
        resList = list(re.findall(ptn, req)[0]);
        for i in range(len(resList)):
          resList[i] = re.sub('MY', 'YOUR', resList[i]);
          resList[i] = re.sub('ME', 'YOU', resList[i]);
          
          pattern = person.getName() + ".*";
          for res in resList:
            pattern += res + ".*";
          
        print(re.sub('<NAME>', person.getName(),re.sub(
                       '<SUBS6>', resList[5], re.sub(
                         '<SUBS5>', resList[4], re.sub(
                           '<SUBS4>', resList[3], re.sub(
                             '<SUBS3>', resList[2], re.sub(
                               '<SUBS2>', resList[1], re.sub(
                                 '<SUBS1>', resList[0], responses[4]
                                )
                              )
                            )
                          )
                        )
                      )
                    ));
        break;
        
