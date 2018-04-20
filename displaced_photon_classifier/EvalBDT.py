#!/usr/bin/env python

import os
import math
import argparse
import itertools
import cPickle as pickle

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve
import ROOT as root
import root_pandas as rp

def get_disc_or_minusone(event, disc_lookup):
    """
    Checks the event ID (run/lumi/event) from the tree (first argument)
    and gets the discriminator value corresponding to the event, 
    if available in the lookup dictionary (second argument).
    Returns -1 if the event is not available.
    """
    e = event.event
    if e < 0:
        e = e + 2**32
    return disc_lookup.get((event.run, event.lumi, e), -1)

def fill_discriminator(oldTree, newTree, disc_lookup, s,
        start=0, end=-1):
    """
    Args:
        oldTree: tree from the input ROOT file
        newTree: clone of the input tree
        disc_lookup: dictionary of the form {(run, lumi, event):disc, ...}
        s: struct linked to discriminator branch in newTree
        fill_all: if True, will fill events even if no BDT value available
    Returns: None

    Fills newTree with all events from oldTree, with discriminator
    values filled from the lookup table where present.
    """
    if end < 0:
        end = oldTree.GetEntries()
    for i in range(start, end):
        oldTree.GetEntry(i)
        if i % 1000 == 0:
            print "Processing event {}".format(i)
        s.disc = get_disc_or_minusone(oldTree, disc_lookup)
        if s.disc > -1:
            newTree.Fill()
        else:
            print "Event fails cut", oldTree.run, oldTree.event, oldTree.lumi


if __name__ == '__main__':
    root.gROOT.SetBatch()
    root.gROOT.Reset()

    parser = argparse.ArgumentParser()
    parser.add_argument('in_name')
    parser.add_argument('--chunk-size', type=int, default=10000,
            help='Number of input events to read per job')
    parser.add_argument('--chunk', type=int, default=0,
            help='Which chunk # to run on (0-indexed)')
    args = parser.parse_args()

    # Get tree for output
    tree_name = 'DelayedPhoton'
    bkgFile = root.TFile(args.in_name)
    bkgTree = bkgFile.Get(tree_name)

    # Define variables to be used for BDT
    variables = ['pho1ecalPFClusterIso', 'pho1trkSumPtHollowConeDR03',
            'pho1PFsumNeutralHadronEt', 'pho1SigmaIetaIeta', 'pho1Smajor',
            'pho1Sminor','pho1R9'] 
    
    # To write out the events with GBM discriminator appended, 
    # we identify the test events by run/lumi/event number.
    id_variables = ['run', 'lumi', 'event']

    # Get ROOT file into pandas
    df_iter = rp.read_root(args.in_name, tree_name,
            columns=variables+id_variables, chunksize=args.chunk_size)
    # root_pandas seems to be set up to read a ROOT file sequentially.
    # Therefore we have to cycle through the file to find the 
    # part we want.
    for i, df in enumerate(df_iter):
        if i == args.chunk:
            break

    # Get min and max entries for this job
    num_entries = bkgTree.GetEntries()
    start_entry = args.chunk * args.chunk_size
    end_entry = min(num_entries,
            (args.chunk + 1) * args.chunk_size)
    if num_entries % args.chunk_size == 0:
        num_chunks = num_entries / args.chunk_size
    else:
        num_chunks = (num_entries / args.chunk_size) + 1
    print "Input ROOT file has {} entries.".format(num_entries)
    print "Will process chunk {} (entries {} - {})".format(
            args.chunk, start_entry, end_entry)

    # Get index of each variable in the DataFrame
    var_indices = [df.columns.get_loc(v) for v in variables] 
    id_var_indices = [df.columns.get_loc(v) for v in id_variables]

    # Get numpy array from pandas data frame
    x = df.values
    x_reduced = x[:,var_indices] # use this for BDT
    x_index = x[:,id_var_indices] # use this to give each event an ID

    # Read in model saved from previous running of BDT
    infile = open("model.pkl","rb")
    model = pickle.load(infile)

    # make prediction for input data
    y_pred = model.predict_proba(x_reduced)[:, 1]

    # Create a lookup table for discriminator value by event number
    disc_lookup = {}
    for disc_val, run, lumi, event in zip(
            y_pred, x_index[:,0], 
            x_index[:,1], x_index[:,2]):
        disc_lookup[(run, lumi, event)] = disc_val

    # Write to a new ROOT file in the local directory
    out_bkg_name = os.path.basename(args.in_name).replace(
            '.root', '_BDT_{}of{}.root'.format(
                args.chunk+1, num_chunks))
    out_bkg = root.TFile(out_bkg_name, 'RECREATE')
    out_bkg_tree = bkgTree.CloneTree(0)

    # This is the boilerplate code for writing something
    # to a ROOT tree from Python.
    root.gROOT.ProcessLine("struct MyStruct{float disc;};")
    from ROOT import MyStruct
    s = MyStruct()
    disc_branch_bkg = out_bkg_tree.Branch(
            'disc', root.AddressOf(s, 'disc'), 'disc/F');

    print "Writing new ROOT background file with discriminator appended"
    fill_discriminator(bkgTree, out_bkg_tree, disc_lookup, s,
            start=start_entry, end=end_entry)

    # Cristian's code uses GetCurrentFile() for this part.
    # I will do that too just in case (paranoia).
    out_bkg.cd()
    out_bkg_tree.GetCurrentFile().Write()
    out_bkg_tree.GetCurrentFile().Close()
