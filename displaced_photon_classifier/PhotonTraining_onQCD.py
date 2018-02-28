#!/usr/bin/env python

#import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
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

root.gROOT.Reset()


#QCD_FileName = '/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/DelayedPhoton_GJets_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'
QCD_FileName = '/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/DelayedPhoton_QCD_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'

test_name = 'MET_only_and_onlyOneTreeWithOneLeaf'

##Define variables to be used

variables = ['pho1ecalPFClusterIso','pho1trkSumPtHollowConeDR03','pho1PFsumNeutralHadronEt','pho1SigmaIetaIeta','pho1Smajor','pho1Sminor','pho1R9'] 


##Getting ROOT files into pandas
df_QCD = rp.read_root(QCD_FileName,
                          'DelayedPhoton', columns=variables)

#getting a numpy array from two pandas data frames
x = df_QCD.values

# fit model
# Read in model saved from previous running of BDT
infile = open("model.pkl","rb")
model = pickle.load(infile)
print( dir(model) )
print model

# make predictions for QCD data
y_pred = model.predict_proba(x)[:, 1]

# make a function to compute the efficiencies
def compute_efficiency(discriminator, cut):
    total = len(discriminator)
    passing = (discriminator > cut).sum()
    return passing * 1.0 / total

# set the cuts to use
cuts1 = np.arange(0, 0.1, 0.00005)
cuts2 = np.arange(0.1, 0.9, 0.001)
cuts3 = np.arange(0.9, 1, 0.00005)
cuts = np.concatenate([cuts1, cuts2, cuts3])

# write cuts and efficiency to file
#with open('roc_cut_eff.txt', 'w') as textfile:
eff_QCD = np.zeros(len(cuts))
for i, cut in enumerate(cuts):
    efficiency = compute_efficiency(y_pred, cut) # compute efficiency at each cut point
    eff_QCD[i] = efficiency


eff_sig = np.zeros(len(cuts))
eff_bkg = np.zeros(len(cuts))

with open('roc_cut_eff.txt', 'r') as textfile:
    for i, line in enumerate(textfile):
        if i==0:
            continue
        splitline = line.split(',')
        cut = float(splitline[0])
        efficiency_sig = float(splitline[1])
        efficiency_bkg = float(splitline[2])
        eff_sig[i-1] = efficiency_sig
        eff_bkg[i-1] = efficiency_bkg


plt.figure()
lw = 2
plt.plot(eff_bkg, eff_sig, color='darkorange',
         lw=lw, label=r'$c\tau$ = 200cm, $\Lambda$ = 250 TeV')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Background (false positives)')
plt.ylabel('Signal (true positives)')
plt.title(r'ROC - GMSB vs. $\gamma$ + Jets')
plt.legend(loc="lower right")
#plt.show()
plt.savefig('GMSB_GJets' + '.png')


plt.figure()
lw = 2
plt.plot(eff_QCD, eff_sig, color='darkorange',
         lw=lw, label=r'$c\tau$ = 200cm, $\Lambda$ = 250 TeV')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Background (false positives)')
plt.ylabel('Signal (true positives)')
plt.title('ROC - GMSB vs. QCD')
plt.legend(loc="lower right")
#plt.show()
plt.savefig('GMSB_QCD' + '.png')


plt.figure()
lw = 2
plt.plot(eff_bkg, eff_QCD, color='darkorange',
         lw=lw, label=r'QCD vs. $\gamma$+jets')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Background (false positives)')
plt.ylabel('Signal (true positives)')
plt.title(r'ROC - QCD vs. $\gamma$ + Jets')
plt.legend(loc="lower right")
#plt.show()
plt.savefig('GJets_QCD' + '.png')


plt.figure()
plt.hist(y_pred, normed=True, bins=50, alpha=0.3)
plt.savefig('mydiscriminator_QCD.png')

with open('roc_QCD_GJets_GMSB.txt', 'w') as textfile:
    textfile.write("cut, efficiency_GMSB, efficiency_GJets, efficiency_QCD \n")
    for i, cut in enumerate(cuts):
        line = "{},{},{},{}\n".format(cut, eff_sig[i], eff_bkg[i], eff_QCD[i])
        textfile.write(line) # save cut and efficiency to file
        print line

