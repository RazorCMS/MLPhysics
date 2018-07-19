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

plotDir = '/afs/cern.ch/user/z/zhicaiz/www/sharebox/DelayedPhoton/10July2018/BDT/'

root.gROOT.ProcessLine("struct MyStruct{float disc;};")
from ROOT import MyStruct

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

# make a function to compute the efficiencies
def compute_efficiency(discriminator, cut):
    total = len(discriminator)
    passing = (discriminator > cut).sum()
    return passing * 1.0 / total

fileLists = {
	'GMSB_L350TeV_Ctau200cm':'/eos/cms/store/group/phys_susy/razor/Run2Analysis/DelayedPhotonAnalysis/2016/V4p1_private_REMINIAOD/skim_BDT_test/GMSB_L350TeV_Ctau200cm_13TeV-pythia8.root',
	'QCD_HT1000to1500':'/eos/cms/store/group/phys_susy/razor/Run2Analysis/DelayedPhotonAnalysis/2016/V4p1_private_REMINIAOD/skim_BDT_test/DelayedPhoton_QCD_HT1000to1500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.root',
	'QCD_HT1500to2000':'/eos/cms/store/group/phys_susy/razor/Run2Analysis/DelayedPhotonAnalysis/2016/V4p1_private_REMINIAOD/skim_BDT_test/DelayedPhoton_QCD_HT1500to2000_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.root',
	'QCD_HT2000toInf':'/eos/cms/store/group/phys_susy/razor/Run2Analysis/DelayedPhotonAnalysis/2016/V4p1_private_REMINIAOD/skim_BDT_test/DelayedPhoton_QCD_HT2000toInf_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.root',
	'QCD_HT200to300':'/eos/cms/store/group/phys_susy/razor/Run2Analysis/DelayedPhotonAnalysis/2016/V4p1_private_REMINIAOD/skim_BDT_test/DelayedPhoton_QCD_HT200to300_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.root',
	'QCD_HT300to500':'/eos/cms/store/group/phys_susy/razor/Run2Analysis/DelayedPhotonAnalysis/2016/V4p1_private_REMINIAOD/skim_BDT_test/DelayedPhoton_QCD_HT300to500_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.root',
	'QCD_HT500to700':'/eos/cms/store/group/phys_susy/razor/Run2Analysis/DelayedPhotonAnalysis/2016/V4p1_private_REMINIAOD/skim_BDT_test/DelayedPhoton_QCD_HT500to700_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.root',
	'QCD_HT700to1000':'/eos/cms/store/group/phys_susy/razor/Run2Analysis/DelayedPhotonAnalysis/2016/V4p1_private_REMINIAOD/skim_BDT_test/DelayedPhoton_QCD_HT700to1000_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.root',
	'GJets_HT100to200':'/eos/cms/store/group/phys_susy/razor/Run2Analysis/DelayedPhotonAnalysis/2016/V4p1_private_REMINIAOD/skim_BDT_test/DelayedPhoton_GJets_HT-100To200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.root',
	'GJets_HT200to400':'/eos/cms/store/group/phys_susy/razor/Run2Analysis/DelayedPhotonAnalysis/2016/V4p1_private_REMINIAOD/skim_BDT_test/DelayedPhoton_GJets_HT-100To200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.root',
	'GJets_HT400to600':'/eos/cms/store/group/phys_susy/razor/Run2Analysis/DelayedPhotonAnalysis/2016/V4p1_private_REMINIAOD/skim_BDT_test/DelayedPhoton_GJets_HT-100To200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.root',
	'GJets_HT40to100':'/eos/cms/store/group/phys_susy/razor/Run2Analysis/DelayedPhotonAnalysis/2016/V4p1_private_REMINIAOD/skim_BDT_test/DelayedPhoton_GJets_HT-100To200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.root',
	'GJets_HT600toInf':'/eos/cms/store/group/phys_susy/razor/Run2Analysis/DelayedPhotonAnalysis/2016/V4p1_private_REMINIAOD/skim_BDT_test/DelayedPhoton_GJets_HT-100To200_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.root',
	'DoubleEG_2016BCDEFGH':'/eos/cms/store/group/phys_susy/razor/Run2Analysis/DelayedPhotonAnalysis/2016/V4p1_private_REMINIAOD/skim_BDT_test/DelayedPhoton_DoubleEG_2016BCDEFGH_GoodLumi_31p1186ifb.root',
}

for label, file_name in fileLists.iteritems():
	testFile = root.TFile(file_name)
	testTree = testFile.Get('DelayedPhoton')
	testNevents = testFile.Get('NEvents')
	test_name = 'Photons_modelused'

	##Define variables to be used

	variables = ['pho1ecalPFClusterIso','pho1trkSumPtHollowConeDR03','pho1PFsumNeutralHadronEt','pho1SigmaIetaIeta','pho1Smajor','pho1Sminor','pho1R9'] 
	# To write out the events with GBM discriminator appended, 
	# we identify the test events by run/lumi/event number.
	id_variables = ['run', 'lumi', 'event']

	##Getting ROOT files into pandas
	df_testSample = rp.read_root(file_name,
				  'DelayedPhoton', columns=variables+id_variables)

	var_indices = [df_testSample.columns.get_loc(v) for v in variables] # get positions of all the variables set above
	id_var_indices = [df_testSample.columns.get_loc(v) for v in id_variables]

	#getting a numpy array from two pandas data frames
	x = df_testSample.values
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
	disc_lookup_testSample = {}
	for disc_val, run, lumi, event in zip(
		y_pred, x_index[:,0], #what is y_test - it is effectively always 0, since only background
		x_index[:,1], x_index[:,2]):
	    disc_lookup_testSample[(run, lumi, event)] = disc_val

	# need to write to a new root file still

	#out_testSample_name = os.path.basename(file_name).replace('.root', '_BDT.root')
	out_testSample_name = file_name.replace('skim_BDT_test', 'skim_withBDT')
	out_testSample = root.TFile(out_testSample_name, 'RECREATE')
	out_testSample_tree = testTree.CloneTree(0)

	# This is the boilerplate code for writing something
	# to a ROOT tree from Python.
	s = MyStruct()
	disc_branch_testSample = out_testSample_tree.Branch('disc', root.AddressOf(s, 'disc'), 'disc/F');

	print "Writing new ROOT background file with discriminator appended"
	fill_discriminator(testTree, out_testSample_tree, disc_lookup_testSample, s)

	# Cristian's code uses GetCurrentFile() for this part.
	# I will do that too just in case (paranoia).

	out_testSample.cd()
	out_testSample_tree.GetCurrentFile().Write()
	testNevents.Write()
	out_testSample_tree.GetCurrentFile().Close()


	plt.figure()
	plt.hist(y_pred, normed=True, bins=50, alpha=0.3)
	plt.savefig(plotDir+label+'_disc.png')
	plt.savefig(plotDir+label+'_disc.pdf')

