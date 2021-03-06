#include <iostream>
#include "WWZAnalysis.hh"

int main()
{

  TFile* fin = new TFile("/Users/cmorgoth/Work/data/WWZanalysis/MC/jobs3/WWZJetsTo4L2Nu_4f_TuneCUETP8M1_13TeV_aMCatNLOFxFx_pythia8_1pb_weighted_leptonBaseline_plus_differentFlavor_ttZ_xgBoost_discriminator.root", "READ");
  //  TFile* fin = new TFile("/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_ZZTo4L_13TeV-amcatnloFXFX-pythia8_1pb_weighted.root", "READ");
  TTree* tree_signal = (TTree*)fin->Get("WWZAnalysis");
  
  WWZAnalysis* wwz_signal = new WWZAnalysis( tree_signal );
  wwz_signal->Loop();


  //For Bkg
  fin = new TFile("/Users/cmorgoth/Work/data/WWZanalysis/MC/jobs3/ttZJets_13TeV_madgraphMLM_1pb_weighted_leptonBaseline_plus_differentFlavor_ttZ_xgBoost_discriminator.root", "READ");
  TTree* tree_bkg = (TTree*)fin->Get("WWZAnalysis");

  WWZAnalysis* wwz_bkg = new WWZAnalysis( tree_bkg );
  wwz_bkg->Loop();
  
  return 0;
}
