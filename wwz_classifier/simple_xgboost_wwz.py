#!/usr/bin/env python

import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from xgboost import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve
import root_pandas as rp
import numpy as np
import pandas as pd
import math
import ROOT as root

#################
##Preliminaries
#################
lumi = 100.#1/fb
pb2fb = 1000.

#####getting selection efficiency###########

#signal
signalFileEffName = '/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_WWZJetsTo4L2Nu_4f_TuneCUETP8M1_13TeV_aMCatNLOFxFx_pythia8_leptonBaseline.root'
signalFileEff = root.TFile(signalFileEffName)
signalTreeEff = signalFileEff.Get('WWZAnalysis')
signalTreeEff.Draw('MET>>tmp1', 'weight*(1)')
signalHistoEff = root.gDirectory.Get('tmp1')
signalNevents = signalFileEff.Get('NEvents')
signalEff = signalHistoEff.Integral()/signalNevents.Integral()

#bkg
bkgFileEffName    = '/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_ZZTo4L_13TeV-amcatnloFXFX-pythia8_leptonBaseline.root'
bkgFileEff = root.TFile(bkgFileEffName)
bkgTreeEff = bkgFileEff.Get('WWZAnalysis')
bkgTreeEff.Draw('MET>>tmp2', 'weight*(1)')
bkgHistoEff = root.gDirectory.Get('tmp2')
bkgNevents = bkgFileEff.Get('NEvents')
bkgEff = bkgHistoEff.Integral()/bkgNevents.Integral()

signalXsec = 0.1651*0.003179651616*pb2fb
bkgXsec    = 1.212*pb2fb

Sqrt_EffTimesXsec = math.sqrt(bkgXsec*bkgEff + signalXsec*signalEff)

print '[INFO]: sqrt(Eff*XSEC): ', Sqrt_EffTimesXsec

root.gROOT.Reset()

#####getting unconditional probabilities###########

#signal
signalFileName = '/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_WWZJetsTo4L2Nu_4f_TuneCUETP8M1_13TeV_aMCatNLOFxFx_pythia8_1pb_weighted_leptonBaseline.root'
signalFile = root.TFile(signalFileName)
signalTree = signalFile.Get('WWZAnalysis')
signalTree.Draw('MET>>tmp3', 'weight*(1)')
signalHisto = root.gDirectory.Get('tmp3')
signalEvents = pb2fb*lumi*signalHisto.Integral()

#bkg
bkgFileName    = '/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_ZZTo4L_13TeV-amcatnloFXFX-pythia8_1pb_weighted_leptonBaseline.root'
bkgFile = root.TFile(bkgFileName)
bkgTree = bkgFile.Get('WWZAnalysis')
bkgTree.Draw('MET>>tmp4', 'weight*(1)')
bkgHisto = root.gDirectory.Get('tmp4')
bkgEvents = pb2fb*lumi*bkgHisto.Integral()

PofS = signalEvents/(signalEvents+bkgEvents)
PofB = 1. - PofS

print '[INFO]: p(S) =', PofS, '; P(B) =', PofB

test_name = 'MET_only_and_onlyOneTreeWithOneLeaf'

##Define variables to be used
#variables = ['MET','METPhi','lep1Pt','lep2Pt','lep3Pt','lep4Pt','NJet20','NJet30','NBJet20','NBJet30','lep1Phi','lep2Phi','lep3Phi','lep4Phi','lep1Eta','lep2Eta','lep3Eta','lep4Eta','ZMass','ZPt','lep3MT','lep4MT','lep34MT','phi0','theta0','phi','theta1','theta2','phiH','minDRJetToLep3','minDRJetToLep4']
#variables = ['MET','lep1Pt','lep2Pt','lep3Pt','lep4Pt','ZMass','lep3Id', 'lep4Id']
variables = ['MET','lep1Pt','lep2Pt','lep3Pt','lep4Pt','ZMass','lep3Id', 'lep4Id','ZPt','lep3MT','lep4MT','lep34MT','phi0','theta0','phi','theta1','theta2','phiH']
#variables = ['MET']

##Getting ROOT files into pandas
#df_signal = rp.read_root('/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_WWZJetsTo4L2Nu_4f_TuneCUETP8M1_13TeV_aMCatNLOFxFx_pythia8.root', 'WWZAnalysis', columns=['MET','lep1Pt','lep2Pt','lep3Pt','lep4Pt'])
#df_bkg = rp.read_root('/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_ZZTo4L_13TeV-amcatnloFXFX-pythia8.root', 'WWZAnalysis', columns=['MET','lep1Pt','lep2Pt','lep3Pt','lep4Pt'])
df_signal = rp.read_root(signalFileEffName,
                             'WWZAnalysis', columns=variables)
df_bkg = rp.read_root(bkgFileEffName,
                          'WWZAnalysis', columns=variables)


##Getting Sample weights
weights_signal = rp.read_root(signalFileEffName,
                                  'WWZAnalysis', columns=['weight'])
weights_bkg = rp.read_root(bkgFileEffName,
                          'WWZAnalysis', columns=['weight'])


## We care about the sign of the sample only, we don't care about the xsec weight
#weights_sign_signal = np.sign(weights_signal)
#weights_sign_bkg = np.sign(weights_bkg)
weights_sign_signal = weights_signal
weights_sign_bkg = weights_bkg

#print weights_sign_signal.head()
#print weights_sign_bkg.head()

##Getting a numpy array out of two pandas data frame
sample_weights = np.concatenate([weights_sign_bkg.values, weights_sign_signal.values])
print sample_weights

#getting a numpy array from two pandas data frames
x = np.concatenate([df_bkg.values,df_signal.values])
#creating numpy array for target variables
y = np.concatenate([np.zeros(len(df_bkg)),
                        np.ones(len(df_signal))])

# split data into train and test sets
seed = 7
test_size = 0.4
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

# fit model no training data
#model = XGBClassifier(max_depth=1, n_estimators=1, gamma=1, silent=True)
#model = XGBClassifier(max_depth=2, gamma=1, silent=True)
#model = XGBClassifier(max_depth=1,n_estimators=1)
model = XGBClassifier()
#model.fit(x_train, y_train, sample_weight=sample_weights)
model.fit(x_train, y_train)

#print( dir(model) )
print model



# make predictions for test data
#y_pred = model.predict(x_test)
y_pred = model.predict_proba(x_test)[:, 1]
predictions = [round(value) for value in y_pred]

print y_pred

# make histogram of discriminator value for signal and bkg
pd.DataFrame({'truth':y_test, 'disc':y_pred}).hist(column='disc', by='truth', bins=50)

print y_pred

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#get roc curve
#roc = roc_curve(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred)

print "False Positive Rate: ", fpr
print "True Positive Rate: ", tpr
#plot roc curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
#plt.show()
plt.savefig('myroc_' + test_name + '.png')


## plot S/sqrt(B)
significance = []
effSignal = []
effBkg = []
print len(fpr)

ctr = 0
for i in range(len(fpr)):
    if fpr[i] > 1e-5 and tpr[i] > 1e-5:
        print fpr[i], tpr[i] 
        #significance.append(math.sqrt(lumi)*4.8742592356*0.006431528796*tpr[i]/math.sqrt(fpr[i]*0.9935684712))
        significance.append(math.sqrt(lumi)*Sqrt_EffTimesXsec*PofS*tpr[i]/math.sqrt(fpr[i]*PofB))
        effSignal.append(tpr[i])
        effBkg.append(fpr[i])
        #print significance[ctr], ' ' , fpr[ctr], ' ', tpr[ctr]
        ctr = ctr + 1


print "signal Eff: ", effSignal        
print "significance:", significance
        
plt.figure()
plt.plot(effSignal, significance, color='darkorange',
         lw=lw, label='s/sqrt(B)')
#plt.plot([0, 1], [0, 10], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 5.01])
plt.xlabel('signal efficiency')
plt.ylabel('s/sqrt(B)')
plt.title('significance')
plt.legend(loc="lower right")
#plt.show()
plt.savefig('significance_' + test_name + '.png')


plt.figure()
plt.plot(effBkg, significance, color='darkorange',
         lw=lw, label='s/sqrt(B)')
#plt.plot([0, 1], [0, 10], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, .2])
plt.ylim([0.0, 5.01])
plt.xlabel('bkg. efficiency')
plt.ylabel('s/sqrt(B)')
plt.title('significance')
plt.legend(loc="lower right")
#plt.show()
plt.savefig('significance_bkg_' + test_name + '.png')



# Plot
plt.figure()
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
#plt.show()
plt.savefig('myImportances_' + test_name + '.png')

plot_tree( model )
plt.savefig('myTree_' + test_name + '.png')

print "MAXIMUM SIGNIFICANCE = ", max(significance)
