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
signalFileEffNames = {
    'Ctau10cm':'/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/GMSB_L250TeV_Ctau10cm_13TeV-pythia8_pho1Pt_EB.root',
    'Ctau50cm':'/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/GMSB_L250TeV_Ctau50cm_13TeV-pythia8_pho1Pt_EB.root',
    'Ctau200cm':'/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/GMSB_L250TeV_Ctau200cm_13TeV-pythia8_pho1Pt_EB.root',
    'Ctau400cm':'/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/GMSB_L250TeV_Ctau400cm_13TeV-pythia8_pho1Pt_EB.root',
    }


signalEffs = {}

for key, fname in signalFileEffNames.iteritems():
    signalFileEff = root.TFile(fname)
    signalTreeEff = signalFileEff.Get('DelayedPhoton')
    hname = 'h'+key
    signalTreeEff.Draw('MET>>'+hname, 'weight*(1)')
    signalHistoEff = root.gDirectory.Get(hname)
    signalNevents = signalFileEff.Get('NEvents')
    signalEffs[key] = signalHistoEff.Integral()/signalNevents.Integral()

#bkg
bkgFileEffName = '/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/DelayedPhoton_GJets_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'
#bkgFileEffName = '/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/DelayedPhoton_QCD_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'
#bkgFileEffName = '/afs/cern.ch/work/g/gkopp/Thesis/MC_Nov24_noIsoCut_barrelskim/DelayedPhoton_GJets_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'
#bkgFileEffName = '/afs/cern.ch/work/g/gkopp/Thesis/MC_Nov24_noIsoCut_barrelskim/DelayedPhoton_QCD_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'

bkgFileEff = root.TFile(bkgFileEffName)
bkgTreeEff = bkgFileEff.Get('DelayedPhoton')
bkgTreeEff.Draw('MET>>tmp2', 'weight*(1)')
bkgHistoEff = root.gDirectory.Get('tmp2')
bkgNevents = bkgFileEff.Get('NEvents')
bkgEff = bkgHistoEff.Integral()/bkgNevents.Integral()

signalXsec = 0.1651*0.003179651616*pb2fb
bkgXsec    = 1.212*pb2fb

Sqrt_EffTimesXsecs = {}

for key, s_eff in signalEffs.iteritems():
    Sqrt_EffTimesXsecs[key] = math.sqrt(bkgXsec*bkgEff + signalXsec*s_eff)
    print '[INFO]: sqrt(Eff*XSEC): ', Sqrt_EffTimesXsecs[key]

root.gROOT.Reset()

#####getting unconditional probabilities###########

#signal
signalFileNames = {
    'Ctau10cm':'/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/GMSB_L250TeV_Ctau10cm_13TeV-pythia8_pho1Pt_EB.root',
    'Ctau50cm':'/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/GMSB_L250TeV_Ctau50cm_13TeV-pythia8_pho1Pt_EB.root',
    'Ctau200cm':'/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/GMSB_L250TeV_Ctau200cm_13TeV-pythia8_pho1Pt_EB.root',
    'Ctau400cm':'/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/GMSB_L250TeV_Ctau400cm_13TeV-pythia8_pho1Pt_EB.root',
    }

signalEvents = {}

for key, fname in signalFileNames.iteritems():
    signalFile = root.TFile(fname)
    signalTree = signalFile.Get('DelayedPhoton')
    hname_2 = 'h'+key
    signalTree.Draw('MET>>'+hname_2, 'weight*(1)')
    signalHisto = root.gDirectory.Get(hname_2)
    signalEvents[key] = pb2fb*lumi*signalHisto.Integral()


#bkg
bkgFileName = '/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/DelayedPhoton_GJets_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'
#bkgFileName = '/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/DelayedPhoton_QCD_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'
#bkgFileName = '/afs/cern.ch/work/g/gkopp/Thesis/MC_Nov24_noIsoCut_barrelskim/DelayedPhoton_GJets_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'
#bkgFileName = '/afs/cern.ch/work/g/gkopp/Thesis/MC_Nov24_noIsoCut_barrelskim/DelayedPhoton_QCD_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'

bkgFile = root.TFile(bkgFileName)
bkgTree = bkgFile.Get('DelayedPhoton')
bkgTree.Draw('MET>>tmp4', 'weight*(1)')
bkgHisto = root.gDirectory.Get('tmp4')
bkgEvents = pb2fb*lumi*bkgHisto.Integral()

PofS = {}
PofB = {}

for key, sig_event in signalEvents.iteritems():
    PofS[key] = sig_event/(sig_event+bkgEvents)
    PofB[key] = 1. - PofS[key]

print '[INFO]: p(S) =', PofS, '; P(B) =', PofB

test_name = 'Photons_GJets_250tev_time'

##Define variables to be used

#variables = ['pho1ecalPFClusterIso','pho1trkSumPtHollowConeDR03','pho1PFsumNeutralHadronEt','pho1SigmaIetaIeta','pho1Smajor','pho1Sminor','pho1R9','pho1SeedTimeRaw', 'pho1HoverE']
variables = ['pho1ecalPFClusterIso','pho1trkSumPtHollowConeDR03','pho1PFsumNeutralHadronEt','pho1SigmaIetaIeta','pho1Smajor','pho1Sminor','pho1R9','pho1SeedTimeRaw']

##Getting ROOT files into pandas
#df_signal = rp.read_root('/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_WWZJetsTo4L2Nu_4f_TuneCUETP8M1_13TeV_aMCatNLOFxFx_pythia8.root', 'WWZAnalysis', columns=['MET','lep1Pt','lep2Pt','lep3Pt','lep4Pt'])
#df_bkg = rp.read_root('/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_ZZTo4L_13TeV-amcatnloFXFX-pythia8.root', 'WWZAnalysis', columns=['MET','lep1Pt','lep2Pt','lep3Pt','lep4Pt'])

df_signals = {}

for key, fname in signalFileNames.iteritems():
    df_signals[key] = rp.read_root(fname,
                            'DelayedPhoton', columns=variables)

df_bkg = rp.read_root(bkgFileName,
                          'DelayedPhoton', columns=variables)

##Getting Sample weights

weights_signals = {}

for key, fname in signalFileEffNames.iteritems():
    weights_signals[key] = rp.read_root(fname,
                                  'DelayedPhoton', columns=['weight'])

weights_bkg = rp.read_root(bkgFileEffName,
                          'DelayedPhoton', columns=['weight'])

## We care about the sign of the sample only, we don't care about the xsec weight
#weights_sign_signal = np.sign(weights_signal)
#weights_sign_bkg = np.sign(weights_bkg)

weights_sign_signal = weights_signals
weights_sign_bkg = weights_bkg

#print weights_sign_signal.head()
#print weights_sign_bkg.head()

##Getting a numpy array out of two pandas data frame

sample_weights = {}

for key, weight in weights_sign_signal.iteritems():
    sample_weights[key] = np.concatenate([weights_sign_bkg.values, weight.values])
#print sample_weights

#getting a numpy array from two pandas data frames
x = {}
y = {}

#creating numpy array for target variables
for key, signal in df_signals.iteritems():
    x[key] = np.concatenate([df_bkg.values,signal.values])
    y[key] = np.concatenate([np.zeros(len(df_bkg)),np.ones(len(signal))])

# split data into train and test sets
seed = 7
test_size = 0.4

x_train = {}
x_test = {}
y_train = {}
y_test = {}

for key, xname in x.iteritems():
    x_train[key], x_test[key], y_train[key], y_test[key] = train_test_split(xname, y[key], test_size=test_size, random_state=seed)

#x_test_barrel = [event for event in x_test if event[8] < 1.5] # test on events only in barrel
#x_test_barrel = x_test[ x_test[:, 8] < 1.5] # test on events only in barrel
#print x_test

# fit model no training data
#model = XGBClassifier(max_depth=8, n_estimators=1000, gamma=1, learning_rate=0.99, silent=True)
#model = XGBClassifier(max_depth=2, gamma=1, silent=True)
#model = XGBClassifier(max_depth=1,n_estimators=1)

print "fitting model"
model = {}     

for key, fname in x_train.iteritems():
    model[key] = XGBClassifier()
    model[key].fit(x_train[key], y_train[key])

#print( dir(model) )
#print model

# make predictions for test data
#y_pred = model.predict(x_test)

y_pred = {}
predictions = {}

for key, fname in model.iteritems():
    y_pred[key] = fname.predict_proba(x_test[key])[:, 1]
    predictions[key] = [round(value) for value in y_pred[key]]


#print y_pred
##########################################################
# make histogram of discriminator value for signal and bkg
##########################################################
#pd.DataFrame({'truth':y_test, 'disc':y_pred}).hist(column='disc', by='truth', bins=50)

y_frame = {}
disc_bkg = {}
disc_signal = {}

for key, fname in y_pred.iteritems():
    y_frame[key] = pd.DataFrame({'truth':y_test[key], 'disc':y_pred[key]})
    
for key, fname in y_frame.iteritems():
    disc_bkg[key] = fname[fname['truth'] == 0]['disc'].values
    disc_signal[key] = fname[fname['truth'] == 1]['disc'].values


plt.figure()
plt.hist(disc_bkg['Ctau200cm'], normed=True, bins=50, alpha=0.3)
plt.hist(disc_signal['Ctau200cm'], normed=True, bins=50, alpha=0.3)
plt.savefig('mydiscriminator.png')
print "disc_bkg: ", disc_bkg['Ctau200cm']
print "disc_signal: ", disc_signal['Ctau200cm']

# evaluate predictions
accuracy = accuracy_score(y_test['Ctau200cm'], predictions['Ctau200cm'])
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#get roc curve
#roc = roc_curve(y_test, y_pred)

fpr = {}
tpr = {}

for key, fname in y_test.iteritems():
    fpr[key], tpr[key], _ = roc_curve(fname, y_pred[key])

#for i in range(len(fpr)):
#    if fpr[i] >0.199 and fpr[i] < 0.201:
#        print "False Positive Rate: ", fpr[i]
#        print "True Positive Rate: ", tpr[i]

#plot roc curve
print "made it to ROC curve"
plt.figure()
lw = 2

#for key, fname in fpr.iteritems():
#    plt.plot(fname, tpr[key], color = 'darkorange', lw=lw, label=key)
# standard order dark orange, red, blue, green

#for ctau 200cm
plt.plot(fpr['Ctau10cm'], tpr['Ctau10cm'], color='darkorange',
         lw=lw, label=r'$c\tau$ = 10 cm')
plt.plot(fpr['Ctau50cm'], tpr['Ctau50cm'], color='red',
         lw=lw, label=r'$c\tau$ = 50 cm')
plt.plot(fpr['Ctau200cm'], tpr['Ctau200cm'], color='blue',
         lw=lw, label=r'$c\tau$ = 200 cm')
plt.plot(fpr['Ctau400cm'], tpr['Ctau400cm'], color='green',
         lw=lw, label=r'$c\tau$ = 400 cm')



plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Background (false positives)')
plt.ylabel('Signal (true positives)')
plt.title(r'ROC - GMSB vs. $\gamma$ + Jets, $\Lambda$ = 250 TeV, 7 variables + ECAL Timing')
plt.legend(loc="lower right")
#plt.show()
plt.savefig('myroc_' + test_name + '.png')


## plot S/sqrt(B)
significance = []
effSignal = []
effBkg = []
#print len(fpr)

# ctr = 0
# for i in range(len(fpr['Ctau10cm'])):
#     if fpr[i] > 1e-5 and tpr[i] > 1e-5:
#         #print fpr[i], tpr[i] 
#         #significance.append(math.sqrt(lumi)*4.8742592356*0.006431528796*tpr[i]/math.sqrt(fpr[i]*0.9935684712))
#         significance.append(math.sqrt(lumi)*Sqrt_EffTimesXsec*PofS*tpr[i]/math.sqrt(fpr[i]*PofB))
#         effSignal.append(tpr[i])
#         effBkg.append(fpr[i])
#         #print significance[ctr], ' ' , fpr[ctr], ' ', tpr[ctr]
#         ctr = ctr + 1


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
plt.bar(range(len(model['Ctau200cm'].feature_importances_)), model['Ctau200cm'].feature_importances_)
plt.show()
plt.savefig('myImportances_' + test_name + '.png')

#plot_tree( model['Ctau200cm'] )
#plt.savefig('myTree_' + test_name + '.png')

print "MAXIMUM SIGNIFICANCE = ", max(significance['Ctau200cm'])


output = open('model.pkl', 'wb')

# Pickle dictionary using protocol 0.
pickle.dump(model, output)
output.close()
