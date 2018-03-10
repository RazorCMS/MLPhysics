#!/usr/bin/env python

#import matplotlib.pyplot as plt
import os
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

def get_disc_or_minusone(event, disc_lookup):
    """
    Checks the event ID (run/lumi/event) from the tree (first argument)
    and gets the discriminator value corresponding to the event, 
    if available in the lookup dictionary (second argument).
    Returns -1 if the event is not available.
    """
    return disc_lookup.get((event.run, event.lumi, event.event), -1)

def fill_discriminator(oldTree, newTree, disc_lookup, s):
    """
    Args:
        oldTree: tree from the input ROOT file
        newTree: clone of the input tree
        disc_lookup: dictionary of the form {(run, lumi, event):disc, ...}
        s: struct linked to discriminator branch in newTree
    Returns: None

    Fills newTree with all events from oldTree, with discriminator
    values filled from the lookup table where present.
    """
    num_entries = oldTree.GetEntries()
    for i in range(num_entries):
        oldTree.GetEntry(i)
        if i % 10000 == 0:
            print "Processing event {} of {}".format(i, num_entries)
        s.disc = get_disc_or_minusone(oldTree, disc_lookup)
        newTree.Fill()

root.gROOT.SetBatch()
root.gROOT.Reset()


#GJets_FileName = '/Users/gillian/Documents/Caltech/Senior/Thesis/DelayedPhotonFiles/MC_Nov24_noIsoCuts_barrelskim/DelayedPhoton_GJets_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'
GJets_FileName = '/afs/cern.ch/work/g/gkopp/public/ThesisROOTfiles/DelayedPhoton_GJets_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'
bkgFile = root.TFile(GJets_FileName)
bkgTree = bkgFile.Get('DelayedPhoton')
bkgNevents = bkgFile.Get('NEvents')
test_name = 'Photons_modelused'

##Define variables to be used

variables = ['pho1ecalPFClusterIso','pho1trkSumPtHollowConeDR03','pho1PFsumNeutralHadronEt','pho1SigmaIetaIeta','pho1Smajor','pho1Sminor','pho1R9'] 
# To write out the events with GBM discriminator appended, 
# we identify the test events by run/lumi/event number.
id_variables = ['run', 'lumi', 'event']

##Getting ROOT files into pandas
df_GJets = rp.read_root(GJets_FileName,
                          'DelayedPhoton', columns=variables+id_variables)

var_indices = [df_GJets.columns.get_loc(v) for v in variables] # get positions of all the variables set above
id_var_indices = [df_GJets.columns.get_loc(v) for v in id_variables]

#getting a numpy array from two pandas data frames
x = df_GJets.values
x_reduced = x[:,var_indices]
x_index = x[:,id_var_indices]

# fit model
# Read in model saved from previous running of BDT
infile = open("model.pkl","rb")
model = pickle.load(infile)
print( dir(model) )
print model

# make predictions for GJets data
y_pred = model.predict_proba(x_reduced)[:, 1]


# looking at line 107 in other file
# dont need to add discriminator value for signal, it is already in the file
# just add for GJets background

# Create a lookup table for discriminator value by event number
disc_lookup_bkg = {}
for disc_val, run, lumi, event in zip(
        y_pred, x_index[:,0], #what is y_test - it is effectively always 0, since only background
        x_index[:,1], x_index[:,2]):
    disc_lookup_bkg[(run, lumi, event)] = disc_val

# need to write to a new root file still

out_bkg_name = os.path.basename(GJets_FileName).replace('.root', '_BDT.root')
out_bkg = root.TFile(out_bkg_name, 'RECREATE')
out_bkg_tree = bkgTree.CloneTree(0)

# This is the boilerplate code for writing something
# to a ROOT tree from Python.
root.gROOT.ProcessLine("struct MyStruct{float disc;};")
from ROOT import MyStruct
s = MyStruct()
disc_branch_bkg = out_bkg_tree.Branch('disc', root.AddressOf(s, 'disc'), 'disc/F');

print "Writing new ROOT background file with discriminator appended"
fill_discriminator(bkgTree, out_bkg_tree, disc_lookup_bkg, s)

# Cristian's code uses GetCurrentFile() for this part.
# I will do that too just in case (paranoia).

out_bkg.cd()
out_bkg_tree.GetCurrentFile().Write()
bkgNevents.Write()
out_bkg_tree.GetCurrentFile().Close()


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
eff_GJets = np.zeros(len(cuts))
for i, cut in enumerate(cuts):
    efficiency = compute_efficiency(y_pred, cut) # compute efficiency at each cut point
    eff_GJets[i] = efficiency


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
plt.title(r'ROC - GMSB vs. QCD')
plt.legend(loc="lower right")
#plt.show()
plt.savefig('GMSB_QCD' + '.png')


plt.figure()
lw = 2
plt.plot(eff_GJets, eff_sig, color='darkorange',
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
plt.plot(eff_bkg, eff_GJets, color='darkorange',
         lw=lw, label=r'$\gamma$+jets vs QCD')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Background (false positives)')
plt.ylabel('Signal (true positives)')
plt.title(r'ROC -  $\gamma$ + Jets vs. QCD')
plt.legend(loc="lower right")
#plt.show()
plt.savefig('GJets_QCD' + '.png')


plt.figure()
plt.hist(y_pred, normed=True, bins=50, alpha=0.3)
plt.savefig('mydiscriminator_GJets.png')

with open('roc_QCD_GJets_GMSB.txt', 'w') as textfile:
    textfile.write("cut, efficiency_GMSB, efficiency_QCD, efficiency_GJets \n")
    for i, cut in enumerate(cuts):
        line = "{},{},{},{}\n".format(cut, eff_sig[i], eff_bkg[i], eff_GJets[i])
        textfile.write(line) # save cut and efficiency to file
        print line

