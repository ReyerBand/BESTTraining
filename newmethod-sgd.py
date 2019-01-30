#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ROOT import TFile
from root_numpy import tree2array
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics, preprocessing
from plot_confusion_matrix import plot_confusion_matrix
import copy


# In[2]:


fileWW = TFile("out_WWall.root", "READ")
fileZZ = TFile("out_ZZall.root", "READ")
fileHH = TFile("out_HHall.root", "READ")
fileTT = TFile("out_TTall.root", "READ")
fileJJ = TFile("out_QCDall.root", "READ")
treeWW = fileWW.Get("jetTree")
treeZZ = fileZZ.Get("jetTree")
treeHH = fileHH.Get("jetTree")
treeTT = fileTT.Get("jetTree")
treeJJ = fileJJ.Get("jetTree")


# In[3]:


vars = []
for branch in treeWW.GetListOfBranches():
    name = branch.GetName()
    if 'Njets' in name:
        continue
    if 'target' in name:
        continue
    if 'NNout' in name:
        continue
    if 'sum' in name:
        continue
    if 'gen' in name:
        continue
    if 'flatten' in name:
        continue
    if 'dist' in name:
        continue
    if 'npv' in name:
        continue
    if 'sorting' in name:
        continue
    if name == 'mass':
        continue
    if name == 'minDist':
        continue
    if 'et' in name:
        continue
    vars.append(name)


# In[4]:


sel = "tau32 < 9999. && et > 500. && et < 2500."
treeVars = vars
arrayWW = tree2array(treeWW, treeVars, sel)
arrayZZ = tree2array(treeZZ, treeVars, sel)
arrayHH = tree2array(treeHH, treeVars, sel)
arrayTT = tree2array(treeTT, treeVars, sel)
arrayJJ = tree2array(treeJJ, treeVars, sel)


# In[5]:


newArrayWW = []
newArrayZZ = []
newArrayHH = []
newArrayTT = []
newArrayJJ = []
for entry in arrayWW[:]:
    a = list(entry)
    newArrayWW.append(a)
for entry in arrayZZ[:]:
    a = list(entry)
    newArrayZZ.append(a)
for entry in arrayHH[:]:
    a = list(entry)
    newArrayHH.append(a)
for entry in arrayTT[:]:
    a = list(entry)
    newArrayTT.append(a)
for entry in arrayJJ[:]:
    a = list(entry)
    newArrayJJ.append(a)
arrayWW = copy.copy(newArrayWW)
arrayZZ = copy.copy(newArrayZZ)
arrayHH = copy.copy(newArrayHH)
arrayTT = copy.copy(newArrayTT)
arrayJJ = copy.copy(newArrayJJ)





# In[6]:


histsWW = np.array(arrayWW).T
histsZZ = np.array(arrayZZ).T
histsHH = np.array(arrayHH).T
histsTT = np.array(arrayTT).T
histsJJ = np.array(arrayJJ).T


for index, hist in enumerate(histsWW):
    plt.figure()
    plt.hist(hist, bins=100, color='g', label='W', histtype='step')
    plt.hist(histsZZ[index], bins=100, color='y', label='Z', histtype='step')
    plt.hist(histsHH[index], bins=100, color='m', label='H', histtype='step')
    plt.hist(histsTT[index], bins=100, color='r', label='t', histtype='step')
    plt.hist(histsJJ[index], bins=100, color='b', label='QCD', histtype='step')
    plt.xlabel(vars[index])
    plt.legend()
    plt.show()
    plt.close()


# In[7]:


#randomize dataset
import random

trainData = []
targetData = []
nEvents = len(newArrayWW) + len(newArrayZZ) + len(newArrayHH) + len(newArrayTT) + len(newArrayJJ)
print nEvents
while nEvents > 0:
    rng = random.randint(0,4)
    if (rng == 0 and len(newArrayJJ) > 0):
        trainData.append(newArrayJJ.pop())
        targetData.append(0)
        nEvents = nEvents -1
    if (rng == 1 and len(newArrayWW) > 0):
        trainData.append(newArrayWW.pop())
        targetData.append(1)
        nEvents = nEvents - 1
    if (rng == 2 and len(newArrayZZ) > 0):
        trainData.append(newArrayZZ.pop())
        targetData.append(2)
        nEvents = nEvents - 1
    if (rng == 3 and len(newArrayHH) > 0):
        trainData.append(newArrayHH.pop())
        targetData.append(3)
        nEvents = nEvents - 1
    if (rng == 4 and len(newArrayTT) > 0):
        trainData.append(newArrayTT.pop())
        targetData.append(4)
        nEvents = nEvents - 1


# In[8]:


#standardize dataset
scaler = preprocessing.StandardScaler().fit(trainData)
trainData = scaler.transform(trainData)
arrayTT = scaler.transform(arrayTT)
arrayWW = scaler.transform(arrayWW)
arrayZZ = scaler.transform(arrayZZ)
arrayHH = scaler.transform(arrayHH)
arrayJJ = scaler.transform(arrayJJ)


# In[20]:


from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss="log", penalty="l2", verbose=True, n_iter=100)
sgd.fit(trainData[:100000], targetData[:100000])


# In[21]:


cm = metrics.confusion_matrix(sgd.predict(trainData[100000:]), targetData[100000:])
plt.figure()
targetNames = ['j', 'W', 'Z', 'H', 't']
plot_confusion_matrix(cm.T, targetNames, normalize=True)
plt.show()


# In[22]:


print sgd.score(trainData[100000:], targetData[100000:])


# In[18]:


probsTT = sgd.predict_proba(arrayTT)
probsWW = sgd.predict_proba(arrayWW)
probsZZ = sgd.predict_proba(arrayZZ)
probsHH = sgd.predict_proba(arrayHH)
probsJJ = sgd.predict_proba(arrayJJ)


# In[19]:



plt.close()
plt.figure()
plt.xlabel('Probability for j Classification')
plt.hist(probsWW.T[0], bins=20, range=(0,1), label='W', histtype='step',normed=True, log=True)
plt.hist(probsZZ.T[0], bins=20, range=(0,1), label='Z', histtype='step',normed=True, log=True)
plt.hist(probsHH.T[0], bins=20, range=(0,1), label='H', histtype='step',normed=True, log=True)
plt.hist(probsTT.T[0], bins=20, range=(0,1), label='t', histtype='step',normed=True, log=True)
plt.hist(probsJJ.T[0], bins=20, range=(0,1), label='j', histtype='step',normed=True, log=True)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0.)
plt.show()

plt.xlabel('Probability for W Classification')
plt.hist(probsWW.T[1], bins=20, range=(0,1), label='W', histtype='step',normed=True, log=True)
plt.hist(probsZZ.T[1], bins=20, range=(0,1), label='Z', histtype='step',normed=True, log=True)
plt.hist(probsHH.T[1], bins=20, range=(0,1), label='H', histtype='step',normed=True, log=True)
plt.hist(probsTT.T[1], bins=20, range=(0,1), label='t', histtype='step',normed=True, log=True)
plt.hist(probsJJ.T[1], bins=20, range=(0,1), label='j', histtype='step',normed=True, log=True)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0.)
plt.show()

plt.xlabel('Probability for Z Classification')
plt.hist(probsWW.T[2], bins=20, range=(0,1), label='W', histtype='step',normed=True, log=True)
plt.hist(probsZZ.T[2], bins=20, range=(0,1), label='Z', histtype='step',normed=True, log=True)
plt.hist(probsHH.T[2], bins=20, range=(0,1), label='H', histtype='step',normed=True, log=True)
plt.hist(probsTT.T[2], bins=20, range=(0,1), label='t', histtype='step',normed=True, log=True)
plt.hist(probsJJ.T[2], bins=20, range=(0,1), label='j', histtype='step',normed=True, log=True)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0.)
plt.show()

plt.xlabel('Probability for H Classification')
plt.hist(probsWW.T[3], bins=20, range=(0,1), label='W', histtype='step',normed=True, log=True)
plt.hist(probsZZ.T[3], bins=20, range=(0,1), label='Z', histtype='step',normed=True, log=True)
plt.hist(probsHH.T[3], bins=20, range=(0,1), label='H', histtype='step',normed=True, log=True)
plt.hist(probsTT.T[3], bins=20, range=(0,1), label='t', histtype='step',normed=True, log=True)
plt.hist(probsJJ.T[3], bins=20, range=(0,1), label='j', histtype='step',normed=True, log=True)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0.)
plt.show()

plt.xlabel('Probability for t Classification')
plt.hist(probsWW.T[4], bins=20, range=(0,1), label='W', histtype='step',normed=True, log=True)
plt.hist(probsZZ.T[4], bins=20, range=(0,1), label='Z', histtype='step',normed=True, log=True)
plt.hist(probsHH.T[4], bins=20, range=(0,1), label='H', histtype='step',normed=True, log=True)
plt.hist(probsTT.T[4], bins=20, range=(0,1), label='t', histtype='step',normed=True, log=True)
plt.hist(probsJJ.T[4], bins=20, range=(0,1), label='j', histtype='step',normed=True, log=True)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0.)
plt.show()


# In[ ]:




