#include <iostream>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TROOT.h>

int main( )
{
  gROOT->Reset();
  TFile *oldfile = new TFile("/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_ZZTo4L_13TeV-amcatnloFXFX-pythia8_1pb_weighted.root");
  TTree *oldtree = (TTree*)oldfile->Get("WWZAnalysis");
  Long64_t nentries = oldtree->GetEntries();

  float lep1Pt, lep2Pt, lep3Pt, lep4Pt;
  oldtree->SetBranchAddress("lep1Pt", &lep1Pt);
  oldtree->SetBranchAddress("lep2Pt", &lep2Pt);
  oldtree->SetBranchAddress("lep3Pt", &lep3Pt);
  oldtree->SetBranchAddress("lep4Pt", &lep4Pt);
  
  //Create a new file + a clone of old tree in new file
  TFile *newfile = new TFile("skimmed.root","recreate");
  TTree *newtree = oldtree->CloneTree(0);
  
  for (Long64_t i=0;i<nentries; i++) {
    oldtree->GetEntry(i);
    if ( !( lep1Pt > 10. && lep2Pt> 10. && lep3Pt > 10. && lep4Pt> 10. ) ) continue;
    
    newtree->Fill();
  }
  newtree->Print();
  newtree->AutoSave();
  delete oldfile;
  delete newfile;
}
