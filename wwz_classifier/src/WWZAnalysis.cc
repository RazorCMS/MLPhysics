#define WWZAnalysis_cxx
#include <iostream>
#include "WWZAnalysis.hh"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLorentzVector.h>
#include <vector>

void WWZAnalysis::Loop()
{
  if (fChain == 0) return;
  float lumi = 100.;
   Long64_t nentries = fChain->GetEntriesFast();
   
   float npassed = 0;
   float  ntotal = 0;
   
   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
     Long64_t ientry = LoadTree(jentry);
     if (ientry < 0) break;
     nb = fChain->GetEntry(jentry);   nbytes += nb;
     // if (Cut(ientry) < 0) continue;
     ntotal += weight;
     
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
     
     //******************************
     //Selection Cuts 
     //******************************
     //4Lepton
     if (!(lep1Pt>10 && lep2Pt>10 && lep3Pt>10 && lep4Pt>10 )) continue;
     
     //leading lepton pt selection
     double leadLeptonPt = 0;
     double subleadLeptonPt = 0;
     std::vector<double> leptonPtVector; 
     leptonPtVector.push_back(lep1Pt);
     leptonPtVector.push_back(lep2Pt);
     leptonPtVector.push_back(lep3Pt);
     leptonPtVector.push_back(lep4Pt);
     std::sort(leptonPtVector.begin(), leptonPtVector.end());   
     //cout << "Check: " << leptonPtVector[0] << " " << leptonPtVector[1] << " " << leptonPtVector[2] << " " << leptonPtVector[3] << "\n";
     leadLeptonPt = leptonPtVector[3];
     subleadLeptonPt = leptonPtVector[2];

     
     if (!(leadLeptonPt > 25 && subleadLeptonPt > 15)) continue;
     
     //ZMass Window
     //if (!(ZMass > 76 && ZMass < 106)) continue;
     
     //Opposite Charge on Lep3 and Lep4
     if ( abs(lep3Id)/lep3Id ==  abs(lep4Id)/lep4Id ) continue;
     
     //2nd Z Veto
     //if ( fabs(vLep34.M() - 91) < 15 ) continue;
     
     //MET 
     //if (!(MET > 50)) continue;
     
     //BJet Veto
     //if (!(NBJet20 == 0)) continue;
     
     //Jet Veto
     //if (!(NJet30 == 0)) continue;
     
     
     //******************************
     //Categories
     //******************************
     //Difference Flavor
     if ( abs(lep3Id) == abs(lep4Id) ) continue;
     
     //Same Flavor
     //if ( abs(lep3Id) != abs(lep4Id) ) continue;
     
     if ( !(lep3PassLooseMVAID==1 && lep4PassLooseMVAID==1) ) continue;
     
     if ( disc_ttZ < 0.9835 ) continue;
     h_disc_ZZ->Fill(disc_ZZ, weight*1000.0*lumi);//100/fb events
     
     npassed += weight;
     
   }

   std::cout << "nevents: " << npassed*1000.0*lumi << std::endl;
   std::cout << "nevents total: " << ntotal*1000.0*lumi << std::endl;
   std::cout << "eff:" <<  npassed/ntotal << std::endl;
}
