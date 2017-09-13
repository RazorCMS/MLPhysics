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


#get main tree
ch = root.TChain("WWZAnalysis")
ch.Add('WWZAnalysis_ZZTo4L_13TeV-amcatnloFXFX-pythia8_skimmed.root')
nEntries = ch.GetEntries()
print "nEntries = ", nEntries
#*****set brances*****
#set branche satus, at first, all off
#event information
#new tree
outFile = "qcd.slim.root" 
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
    s.disc = 666
    ch_new.Fill()
ch_new.Print()
# use GetCurrentFile just in case we went over the
# (customizable) maximum file size
ch_new.GetCurrentFile().Write()
ch_new.GetCurrentFile().Close()
