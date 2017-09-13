#include <iostream>
#include <algorithm>
#include <stdlib.h>
//ROOT INCLUDES
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TROOT.h>
#include <TLorentzVector.h>
#include <TString.h>


int main( )
{
  gROOT->Reset();
  TString sampleName = "/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_ttZJets_13TeV_madgraphMLM_1pb_weighted";
  TFile *oldfile = new TFile(sampleName+".root");

  TTree *oldtree = (TTree*)oldfile->Get("WWZAnalysis");
  TH1F* nevents = (TH1F*)oldfile->Get("NEvents");
  Long64_t nentries = oldtree->GetEntries();

  float lep1Pt, lep2Pt, lep3Pt, lep4Pt;
  float lep1Phi, lep2Phi, lep3Phi, lep4Phi;
  float lep1Eta, lep2Eta, lep3Eta, lep4Eta;
  int lep1Id, lep2Id, lep3Id, lep4Id;
  
  oldtree->SetBranchAddress("lep1Pt", &lep1Pt);
  oldtree->SetBranchAddress("lep2Pt", &lep2Pt);
  oldtree->SetBranchAddress("lep3Pt", &lep3Pt);
  oldtree->SetBranchAddress("lep4Pt", &lep4Pt);

  oldtree->SetBranchAddress("lep1Phi", &lep1Phi);
  oldtree->SetBranchAddress("lep2Phi", &lep2Phi);
  oldtree->SetBranchAddress("lep3Phi", &lep3Phi);
  oldtree->SetBranchAddress("lep4Phi", &lep4Phi);

  
  oldtree->SetBranchAddress("lep1Eta", &lep1Eta);
  oldtree->SetBranchAddress("lep2Eta", &lep2Eta);
  oldtree->SetBranchAddress("lep3Eta", &lep3Eta);
  oldtree->SetBranchAddress("lep4Eta", &lep4Eta);

  oldtree->SetBranchAddress("lep1Id", &lep1Id);
  oldtree->SetBranchAddress("lep2Id", &lep2Id);
  oldtree->SetBranchAddress("lep3Id", &lep3Id);
  oldtree->SetBranchAddress("lep4Id", &lep4Id);

  //Create a new file + a clone of old tree in new file
  TFile *newfile = new TFile(sampleName+"_leptonBaseline_plus_differentFlavor.root","recreate");
  TTree *newtree = oldtree->CloneTree(0);
  
  for (Long64_t i=0;i<nentries; i++) {
    oldtree->GetEntry(i);
    if ( !( lep1Pt > 10. && lep2Pt> 10. && lep3Pt > 10. && lep4Pt> 10. ) ) continue;
    
     //******************************
     //Lorentz Vectors
     //******************************
     TLorentzVector vLep1;
     double lep1Mass = 0;
     if (abs(lep1Id) == 11) lep1Mass = 0.000511;
     else if (abs(lep1Id) == 13) lep1Mass = 0.1057;
     vLep1.SetPtEtaPhiM(lep1Pt, lep1Eta, lep1Phi,lep1Mass);
     
     TLorentzVector vLep2;
     double lep2Mass = 0;
     if (abs(lep2Id) == 11) lep2Mass = 0.000511;
     else if (abs(lep2Id) == 13) lep2Mass = 0.1057;
     vLep2.SetPtEtaPhiM(lep2Pt, lep2Eta, lep2Phi,lep2Mass);
     
     TLorentzVector vLep3;
     double lep3Mass = 0;
     if (abs(lep3Id) == 11) lep3Mass = 0.000511;
     else if (abs(lep3Id) == 13) lep3Mass = 0.1057;
     vLep3.SetPtEtaPhiM(lep3Pt, lep3Eta, lep3Phi,lep3Mass);
     
     TLorentzVector vLep4;
     double lep4Mass = 0;
     if (abs(lep4Id) == 11) lep4Mass = 0.000511;
     else if (abs(lep4Id) == 13) lep4Mass = 0.1057;
     vLep4.SetPtEtaPhiM(lep4Pt, lep4Eta, lep4Phi,lep4Mass);
     
     TLorentzVector ZCandidate = vLep1+vLep2;
     TLorentzVector v4L = vLep1+vLep2+vLep3+vLep4;
     TLorentzVector vLep34 = vLep3+vLep4;

     auto ptOrder = [](auto a, auto b) { return a > b; };
     //leading lepton pt selection
     std::vector<double> leptonPtVector; 
     leptonPtVector.push_back(lep1Pt);
     leptonPtVector.push_back(lep2Pt);
     leptonPtVector.push_back(lep3Pt);
     leptonPtVector.push_back(lep4Pt);
     sort(leptonPtVector.begin(), leptonPtVector.end(), ptOrder);
     double leadLeptonPt    = leptonPtVector.at(0);
     double subleadLeptonPt = leptonPtVector.at(1);

     if ( !(leadLeptonPt > 25. && subleadLeptonPt > 15.) ) continue;
     //******************************
     //Categories
     //******************************
     //Difference Flavor
     if ( abs(lep3Id) == abs(lep4Id) ) continue;
     
     //Same Flavor
     //if ( abs(lep3Id) != abs(lep4Id) ) continue;
     
     newtree->Fill();
  }
  newtree->Print();
  newtree->AutoSave();
  nevents->Write();
  delete oldfile;
  delete newfile;
}
