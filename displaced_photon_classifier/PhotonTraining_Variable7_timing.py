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
import cPickle as pickle
import ROOT as root

#################
##Preliminaries
#################
lumi = 100.#1/fb
pb2fb = 1000.

#####getting selection efficiency###########

#signal
signalFileEffName = '/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/GMSB_L250TeV_Ctau200cm_13TeV-pythia8_pho1Pt_EB.root'
#signalFileEffName = '/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/DelayedPhoton_GJets_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'
#signalFileEffName = '/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/DelayedPhoton_QCD_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'

signalFileEff = root.TFile(signalFileEffName)
signalTreeEff = signalFileEff.Get('DelayedPhoton')
signalTreeEff.Draw('MET>>tmp1', 'weight*(1)')
signalHistoEff = root.gDirectory.Get('tmp1')
signalNevents = signalFileEff.Get('NEvents')
signalEff = signalHistoEff.Integral()/signalNevents.Integral()

#bkg
#bkgFileEffName = '/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/DelayedPhoton_GJets_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'
bkgFileEffName = '/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/DelayedPhoton_QCD_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'

bkgFileEff = root.TFile(bkgFileEffName)
bkgTreeEff = bkgFileEff.Get('DelayedPhoton')
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
signalFileName = '/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/GMSB_L250TeV_Ctau200cm_13TeV-pythia8_pho1Pt_EB.root'
#signalFileName = '/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/DelayedPhoton_GJets_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'
#signalFileName = '/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/DelayedPhoton_QCD_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'

signalFile = root.TFile(signalFileName)
signalTree = signalFile.Get('DelayedPhoton')
signalTree.Draw('MET>>tmp3', 'weight*(1)')
signalHisto = root.gDirectory.Get('tmp3')
signalEvents = pb2fb*lumi*signalHisto.Integral()


#bkg
#bkgFileName = '/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/DelayedPhoton_GJets_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'
bkgFileName = '/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/DelayedPhoton_QCD_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'


bkgFile = root.TFile(bkgFileName)
bkgTree = bkgFile.Get('DelayedPhoton')
bkgTree.Draw('MET>>tmp4', 'weight*(1)')
bkgHisto = root.gDirectory.Get('tmp4')
bkgEvents = pb2fb*lumi*bkgHisto.Integral()

PofS = signalEvents/(signalEvents+bkgEvents)
PofB = 1. - PofS

print '[INFO]: p(S) =', PofS, '; P(B) =', PofB

test_name = 'Photons_variable_overlay'

##Define variables to be used

#variables = ['pho1ecalPFClusterIso','pho1trkSumPtHollowConeDR03','pho1PFsumNeutralHadronEt','pho1HoverE','pho1SigmaIetaIeta','pho1Smajor','pho1Sminor','pho1R9'] #for signal
#variables = ['pho1PFsumNeutralHadronEt','pho1PFsumChargedHadronPt','pho1PFsumPhotonEt','pho1HoverE','pho1SigmaIetaIeta','pho1Smajor','pho1Sminor','pho1R9'] #for QCD and GJets

variables = ['pho1ecalPFClusterIso','pho1trkSumPtHollowConeDR03','pho1PFsumNeutralHadronEt','pho1SigmaIetaIeta','pho1Smajor','pho1Sminor','pho1R9']
variables2 = ['pho1ecalPFClusterIso','pho1trkSumPtHollowConeDR03','pho1PFsumNeutralHadronEt','pho1SigmaIetaIeta','pho1Smajor','pho1Sminor','pho1R9','pho1SeedTimeRaw']

##Getting ROOT files into pandas
#df_signal = rp.read_root('/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_WWZJetsTo4L2Nu_4f_TuneCUETP8M1_13TeV_aMCatNLOFxFx_pythia8.root', 'WWZAnalysis', columns=['MET','lep1Pt','lep2Pt','lep3Pt','lep4Pt'])
#df_bkg = rp.read_root('/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_ZZTo4L_13TeV-amcatnloFXFX-pythia8.root', 'WWZAnalysis', columns=['MET','lep1Pt','lep2Pt','lep3Pt','lep4Pt'])
df_signal = rp.read_root(signalFileName,
                             'DelayedPhoton', columns=variables)
df_bkg = rp.read_root(bkgFileName,
                          'DelayedPhoton', columns=variables)
df_signal2 = rp.read_root(signalFileName,
                             'DelayedPhoton', columns=variables2)
df_bkg2 = rp.read_root(bkgFileName,
                          'DelayedPhoton', columns=variables2)



#getting a numpy array from two pandas data frames
x = np.concatenate([df_bkg.values,df_signal.values])
#creating numpy array for target variables
y = np.concatenate([np.zeros(len(df_bkg)),
                        np.ones(len(df_signal))])
x2 = np.concatenate([df_bkg2.values,df_signal2.values])
y2 = np.concatenate([np.zeros(len(df_bkg2)),
                        np.ones(len(df_signal2))])

# split data into train and test sets
seed = 7
test_size = 0.4
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=test_size, random_state=seed)

#x_test_barrel = [event for event in x_test if event[8] < 1.5] # test on events only in barrel
#x_test_barrel = x_test[ x_test[:, 8] < 1.5] # test on events only in barrel
#print x_test

# fit model no training data
#model = XGBClassifier(max_depth=8, n_estimators=1000, gamma=1, learning_rate=0.99, silent=True)
#model = XGBClassifier(max_depth=2, gamma=1, silent=True)
#model = XGBClassifier(max_depth=1,n_estimators=1)
model = XGBClassifier()
model2 = XGBClassifier()
#model.fit(x_train, y_train, sample_weight=sample_weights)
model.fit(x_train, y_train)
model2.fit(x_train2, y_train2)

#print( dir(model) )
#print model


# make predictions for test data
#y_pred = model.predict(x_test)
y_pred = model.predict_proba(x_test)[:, 1]
predictions = [round(value) for value in y_pred]
y_pred2 = model2.predict_proba(x_test2)[:, 1]
predictions2 = [round(value) for value in y_pred2]

#print y_pred
##########################################################
# make histogram of discriminator value for signal and bkg
##########################################################
#pd.DataFrame({'truth':y_test, 'disc':y_pred}).hist(column='disc', by='truth', bins=50)
y_frame = pd.DataFrame({'truth':y_test, 'disc':y_pred})
disc_bkg    = y_frame[y_frame['truth'] == 0]['disc'].values
disc_signal = y_frame[y_frame['truth'] == 1]['disc'].values
plt.figure()
plt.hist(disc_bkg, normed=True, bins=50, alpha=0.3)
plt.hist(disc_signal, normed=True, bins=50, alpha=0.3)
plt.savefig('mydiscriminator.png')
print "disc_bkg: ", disc_bkg
print "disc_signal: ", disc_signal

y_frame2 = pd.DataFrame({'truth':y_test2, 'disc':y_pred2})
disc_bkg2    = y_frame2[y_frame2['truth'] == 0]['disc'].values
disc_signal2 = y_frame2[y_frame2['truth'] == 1]['disc'].values
plt.figure()
plt.hist(disc_bkg2, normed=True, bins=50, alpha=0.3)
plt.hist(disc_signal2, normed=True, bins=50, alpha=0.3)
plt.savefig('mydiscriminator2.png')
print "disc_bkg2: ", disc_bkg2
print "disc_signal2: ", disc_signal2

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#get roc curve
#roc = roc_curve(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred)
fpr2, tpr2, _ = roc_curve(y_test2, y_pred2)

#for i in range(len(fpr)):
#    if fpr[i] >0.199 and fpr[i] < 0.201:
#        print "False Positive Rate: ", fpr[i]
#        print "True Positive Rate: ", tpr[i]

#plot roc curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='7 variables')
plt.plot(fpr2, tpr2, color='blue',
         lw=lw, label='7 variables and ECAL timing')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Background (false positives)')
plt.ylabel('Signal (true positives)')
#plt.title(r'ROC - GMSB ($c\tau$ = 200cm, $\Lambda$ = 250 TeV) vs. $\gamma$ + Jets')
plt.title(r'ROC - GMSB ($c\tau$ = 200cm, $\Lambda$ = 250 TeV) vs. QCD')
plt.legend(loc="lower right")
#plt.show()
plt.savefig('myroc_' + test_name + '.png')


## plot S/sqrt(B)
significance = []
effSignal = []
effBkg = []
#print len(fpr)

ctr = 0
for i in range(len(fpr)):
    if fpr[i] > 1e-5 and tpr[i] > 1e-5:
        #print fpr[i], tpr[i] 
        #significance.append(math.sqrt(lumi)*4.8742592356*0.006431528796*tpr[i]/math.sqrt(fpr[i]*0.9935684712))
        significance.append(math.sqrt(lumi)*Sqrt_EffTimesXsec*PofS*tpr[i]/math.sqrt(fpr[i]*PofB))
        effSignal.append(tpr[i])
        effBkg.append(fpr[i])
        #print significance[ctr], ' ' , fpr[ctr], ' ', tpr[ctr]
        ctr = ctr + 1


#print "signal Eff: ", effSignal        
#print "significance:", significance
        
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
plt.bar(range(len(model2.feature_importances_)), model2.feature_importances_)
#plt.show()
plt.savefig('myImportances_' + test_name + '.png')

plot_tree( model )
plt.savefig('myTree_' + test_name + '.png')

print "MAXIMUM SIGNIFICANCE = ", max(significance)


output = open('model.pkl', 'wb')

# Pickle dictionary using protocol 0.
pickle.dump(model, output)
output.close()
