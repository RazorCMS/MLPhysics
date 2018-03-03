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

#signal
signalFileName = '/afs/cern.ch/work/g/gkopp/public/ThesisROOTfiles/GMSB_L250TeV_Ctau200cm_13TeV-pythia8_skimmedpho1Pt_EB.root'
signalFile = root.TFile(signalFileName)
signalTree = signalFile.Get('DelayedPhoton')
signalNevents = signalFile.Get('NEvents')

#bkg
bkgFileName = '/afs/cern.ch/work/g/gkopp/public/ThesisROOTfiles/DelayedPhoton_QCD_HTall_TuneCUETP8M1_13TeV-madgraphMLM-pythia8_skimmed_skimmedpho1Pt_EB.root'
bkgFile = root.TFile(bkgFileName)
bkgTree = bkgFile.Get('DelayedPhoton')
bkgNevents = bkgFile.Get('NEvents')

test_name = 'Photons'

##Define variables to be used
#variables = ['pho1Sminor','pho1SeedTimeRaw'] #use for 2 var testing
variables = ['pho1ecalPFClusterIso','pho1trkSumPtHollowConeDR03','pho1PFsumNeutralHadronEt','pho1SigmaIetaIeta','pho1Smajor','pho1Sminor','pho1R9'] 

# To write out the events with GBM discriminator appended, 
# we identify the test events by run/lumi/event number.
id_variables = ['run', 'lumi', 'event']

##Getting ROOT files into pandas
df_signal = rp.read_root(signalFileName,
                             'DelayedPhoton', columns=variables+id_variables)
df_bkg = rp.read_root(bkgFileName,
                          'DelayedPhoton', columns=variables+id_variables)

var_indices = [df_signal.columns.get_loc(v) for v in variables] # get positions of all the variables set above
id_var_indices = [df_signal.columns.get_loc(v) for v in id_variables]

#Getting a numpy array from two pandas data frames
x = np.concatenate([df_bkg.values,df_signal.values])
#Creating numpy array for target variables
y = np.concatenate([np.zeros(len(df_bkg)),
                        np.ones(len(df_signal))]) # zero if bkg, 1 if signal

# split data into train and test sets
seed = 7
test_size = 0.4
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

# For training we ignore the columns with the event ID information
x_train_reduced = x_train[:,var_indices]
x_test_reduced = x_test[:,var_indices]

x_test_index = x_test[:,id_var_indices]

# fit model no training data
model = XGBClassifier()
print model
model.fit(x_train_reduced, y_train)

# make predictions for test data
y_pred = model.predict_proba(x_test_reduced)[:, 1]

# Create a lookup table for discriminator value by event number
disc_lookup_signal = {}
disc_lookup_bkg = {}
for disc_val, y_val, run, lumi, event in zip(
        y_pred, y_test, x_test_index[:,0],
        x_test_index[:,1], x_test_index[:,2]):
    if y_val == 1:
        disc_lookup_signal[(run, lumi, event)] = disc_val
    elif y_val == 0:
        disc_lookup_bkg[(run, lumi, event)] = disc_val

# We write out the signal and background events to a new ROOT file.
# For events in the test set, we append the GBM discriminator.  
# For events in the train set, we set the discriminator to -1
# to avoid accidentally reusing the training set in the analysis.
out_signal_name = os.path.basename(signalFileName).replace('.root', '_BDT.root')
out_signal = root.TFile(out_signal_name, 'RECREATE')
out_signal_tree = signalTree.CloneTree(0)

out_bkg_name = os.path.basename(bkgFileName).replace('.root', '_BDT.root')
out_bkg = root.TFile(out_bkg_name, 'RECREATE')
out_bkg_tree = bkgTree.CloneTree(0)

# This is the boilerplate code for writing something
# to a ROOT tree from Python.
root.gROOT.ProcessLine("struct MyStruct{float disc;};")
from ROOT import MyStruct
s = MyStruct()
disc_branch_sig = out_signal_tree.Branch('disc', root.AddressOf(s, 'disc'), 'disc/F');
disc_branch_bkg = out_bkg_tree.Branch('disc', root.AddressOf(s, 'disc'), 'disc/F');

print "Writing new ROOT signal file with discriminator appended"
fill_discriminator(signalTree, out_signal_tree, disc_lookup_signal, s)
print "Writing new ROOT background file with discriminator appended"
fill_discriminator(bkgTree, out_bkg_tree, disc_lookup_bkg, s)

# Cristian's code uses GetCurrentFile() for this part.
# I will do that too just in case (paranoia).
out_signal.cd()
out_signal_tree.GetCurrentFile().Write()
signalNevents.Write()
out_signal_tree.GetCurrentFile().Close()

out_bkg.cd()
out_bkg_tree.GetCurrentFile().Write()
bkgNevents.Write()
out_bkg_tree.GetCurrentFile().Close()

# make a function to compute the efficiencies, use y_test for signal, 1-y_test for bkg
def compute_efficiency(discriminator, cut, SvsBarray):
    total = (SvsBarray).sum()
    passing = ((discriminator > cut) * SvsBarray).sum() # gives signal and passing cuts
    return passing * 1.0 / total

# set the cuts to use
cuts1 = np.arange(0, 0.1, 0.00005)
cuts2 = np.arange(0.1, 0.9, 0.001)
cuts3 = np.arange(0.9, 1, 0.00005)
cuts = np.concatenate([cuts1, cuts2, cuts3])

# write cuts and efficiency to file
with open('roc_cut_eff.txt', 'w') as textfile:
    textfile.write("cut, efficiency_sig, efficiency_bkg \n")
    for cut in cuts:
        efficiency_sig = compute_efficiency(y_pred, cut, y_test) # compute efficiency at each cut point
        efficiency_bkg = compute_efficiency(y_pred, cut, 1-y_test) # compute efficiency at each cut point
        line = "{},{},{}\n".format(cut, efficiency_sig, efficiency_bkg)
        textfile.write(line) # save cut and efficiency to file

##########################################################
# make histogram of discriminator value for signal and bkg
##########################################################
y_frame = pd.DataFrame({'truth':y_test, 'disc':y_pred})
disc_bkg    = y_frame[y_frame['truth'] == 0]['disc'].values
disc_signal = y_frame[y_frame['truth'] == 1]['disc'].values
plt.figure()
plt.hist(disc_bkg, normed=True, bins=50, alpha=0.3)
plt.hist(disc_signal, normed=True, bins=50, alpha=0.3)
plt.savefig('mydiscriminator.png')

#get roc curve
fpr, tpr, _ = roc_curve(y_test, y_pred)

#plot roc curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label=r'$\gamma$+jets vs QCD')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Background (false positives)')
plt.ylabel('Signal (true positives)')
plt.title(r'ROC - GMSB vs. $\gamma$ + Jets, $\Lambda$ = 250 TeV, $c\tau$ = 200 cm')
plt.legend(loc="lower right")
plt.savefig('myroc_' + test_name + '.png')

## plot S/sqrt(B)
significance = []
effSignal = []
effBkg = []

# Plot
plt.figure()
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.savefig('myImportances_' + test_name + '.png')

output = open('model.pkl', 'wb')

# Pickle dictionary using protocol 0.
pickle.dump(model, output)
output.close()
