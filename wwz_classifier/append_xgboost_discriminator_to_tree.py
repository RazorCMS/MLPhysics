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

root.gROOT.Reset()

#####getting unconditional probabilities###########

#signal
FileName = '/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_WWZJetsTo4L2Nu_4f_TuneCUETP8M1_13TeV_aMCatNLOFxFx_pythia8_1pb_weighted_leptonBaseline.root'
File = root.TFile(FileName)
Tree = File.Get('WWZAnalysis')

test_name = 'ReadingXgBoostModel'

##Define variables to be used
variables = ['MET','lep1Pt','lep2Pt','lep3Pt','lep4Pt','ZMass','lep3Id', 'lep4Id','ZPt','lep3MT','lep4MT','lep34MT','phi0','theta0','phi','theta1','theta2','phiH']


##Getting ROOT files into pandas
df = rp.read_root(FileName, 'WWZAnalysis', columns=variables)


#getting a numpy array from two pandas data frames
x_test = df.values
#creating numpy array for target variables
y_test = np.zeros(len(df))


############################
# get model from file
############################
with open('model.pkl', 'rb') as pkl_file:
    model = pickle.load(pkl_file)


# make predictions for test data
y_pred = model.predict_proba(x_test)[:, 1]
predictions = [round(value) for value in y_pred]

#print y_pred
##########################################################
# make histogram of discriminator value for signal and bkg
##########################################################
y_frame = pd.DataFrame({'truth':y_test, 'disc':y_pred})
disc    = y_frame[y_frame['truth'] == 0]['disc'].values
plt.figure()
plt.hist(disc, normed=True, bins=50, alpha=0.3)
plt.savefig('mydiscriminator_' + test_name + '.png')
print "disc_bkg: ", disc
#print y_pred

#############################################
##Creating a new TTree with the discriminator
#############################################

ch = root.TChain("WWZAnalysis")
ch.Add(FileName)
nEntries = ch.GetEntries()
print "nEntries = ", nEntries
#*****set brances*****
#set branche satus, at first, all off
#event information
#new tree
outFile = "slim.root" 
newFile = root.TFile(outFile,"RECREATE") 
ch_new = ch.CloneTree(0)

root.gROOT.ProcessLine("struct MyStruct{float disc;};")

from ROOT import MyStruct

# Create branches in the tree
s = MyStruct()

bpt = ch_new.Branch('disc',root.AddressOf(s,'disc'),'disc/F');

for i in range(nEntries):
    ch.GetEntry(i)
    if i%10000==0:
        print "Processing event nr. %i of %i" % (i,nEntries)
    s.disc = disc[i]
    ch_new.Fill()
ch_new.Print()
# use GetCurrentFile just in case we went over the
# (customizable) maximum file size
ch_new.GetCurrentFile().Write()
ch_new.GetCurrentFile().Close()
