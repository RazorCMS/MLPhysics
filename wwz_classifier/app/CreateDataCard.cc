#include <iostream>
#include <fstream>
#include "WWZAnalysis.hh"

int main()
{


  int nbins = 50;
  float x_low  = 0.0;
  float x_high = 1.0;
  
  TFile* fin = new TFile("/Users/cmorgoth/Work/data/WWZanalysis/MC/jobs3/WWZJetsTo4L2Nu_4f_TuneCUETP8M1_13TeV_aMCatNLOFxFx_pythia8_1pb_weighted_leptonBaseline_plus_differentFlavor_ttZDiscriminator_ZZDiscriminator.root", "READ");
  TTree* tree_signal = (TTree*)fin->Get("WWZAnalysis");
  
  WWZAnalysis* wwz_signal = new WWZAnalysis( tree_signal );
  wwz_signal->h_disc_ZZ = new TH1F("h_signal", "h_signal", nbins, x_low, x_high);
  wwz_signal->Loop();


  //For Bkg ZZ
  fin = new TFile("/Users/cmorgoth/Work/data/WWZanalysis/MC/jobs3/ZZTo4L_13TeV_powheg_pythia8_1pb_weighted_leptonBaseline_plus_differentFlavor_ttZDiscriminator_ZZDiscriminator.root", "READ");
  TTree* tree_ZZ = (TTree*)fin->Get("WWZAnalysis");

  WWZAnalysis* wwz_ZZ = new WWZAnalysis( tree_ZZ );
  wwz_ZZ->h_disc_ZZ = new TH1F("h_ZZ", "h_ZZ", nbins, x_low, x_high);
  wwz_ZZ->Loop();


  //For Bkg ttZ
  fin = new TFile("/Users/cmorgoth/Work/data/WWZanalysis/MC/jobs3/ttZJets_13TeV_madgraphMLM_1pb_weighted_leptonBaseline_plus_differentFlavor_ttZDiscriminator_ZZDiscriminator.root", "READ");
  TTree* tree_ttZ = (TTree*)fin->Get("WWZAnalysis");

  WWZAnalysis* wwz_ttZ = new WWZAnalysis( tree_ttZ );
  wwz_ttZ->h_disc_ZZ = new TH1F("h_ttZ", "h_ttZ", nbins, x_low, x_high);
  wwz_ttZ->Loop();

  //For Bkg WZ
  fin = new TFile("/Users/cmorgoth/Work/data/WWZanalysis/MC/jobs3/WZTo3LNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8_ALL_1pb_weighted_leptonBaseline_plus_differentFlavor_ttZDiscriminator_ZZDiscriminator.root", "READ");
  //fin = new TFile("/Users/cmorgoth/Work/data/WWZanalysis/MC/jobs3/WZTo3LNu_mllmin01_13TeV-powheg-pythia8_ext1_ALL_1pb_weighted_leptonBaseline_plus_differentFlavor_ttZDiscriminator_ZZDiscriminator.root", "READ");
  TTree* tree_WZ = (TTree*)fin->Get("WWZAnalysis");

  WWZAnalysis* wwz_WZ = new WWZAnalysis( tree_WZ );
  wwz_WZ->h_disc_ZZ = new TH1F("h_WZ", "h_WZ", nbins, x_low, x_high);
  wwz_WZ->Loop();

  
  TH1F* h_data = new TH1F("data","data", nbins, x_low, x_high);
  h_data->Add(wwz_ZZ->h_disc_ZZ);
  h_data->Add(wwz_ttZ->h_disc_ZZ);
  h_data->Add(wwz_signal->h_disc_ZZ);
  h_data->Add(wwz_WZ->h_disc_ZZ);

  std::ofstream ofs ("datacards.txt", std::ofstream::out);
  ofs << "imax 1\n";
  ofs << "jmax 3\n";
  //ofs << "jmax 2\n";
  ofs << "kmax *\n";
  ofs << "---------------\n";
  ofs << "shapes * * dataCardShapes.root $PROCESS $PROCESS_$SYSTEMATIC\n";
  ofs << "---------------\n";
  ofs << "bin bin1\n";
  ofs << "observation " << h_data->Integral() << std::endl;
  ofs << "------------------------------\n";

  ofs << "bin             bin1           bin1           bin1            bin1\n";
  ofs << "process         WWZ_signal     WWZ_bkg_ZZ     WWZ_bkg_ttZ     WWZ_bkg_WZ\n";
  ofs << "process         0              1              2               3\n";
  ofs << "rate\t\t" <<  wwz_signal->h_disc_ZZ->Integral() << "\t\t" << wwz_ZZ->h_disc_ZZ->Integral() << "\t\t" << wwz_ttZ->h_disc_ZZ->Integral() << "\t\t" << wwz_WZ->h_disc_ZZ->Integral() <<"\n";
  
  /*
  ofs << "bin             bin1           bin1           bin1\n";
  ofs << "process         WWZ_signal     WWZ_bkg_ZZ     WWZ_bkg_ttZ\n";
  ofs << "process         0              1              2\n";
  ofs << "rate\t\t" <<  wwz_signal->h_disc_ZZ->Integral() << "\t\t" << wwz_ZZ->h_disc_ZZ->Integral() << "\t\t" << wwz_ttZ->h_disc_ZZ->Integral() << "\n";
  */
  ofs << "--------------------------------\n";
  //ofs << "lumi     lnN    1.10 \t\t 1.10 \t\t 1.10 \t\t 1.10\n";
  //ofs << "bgnorm   lnN    1.00 \t\t 1.30 \t\t 1.25 \t\t 1.30\n";
  ofs << "lumi     lnN    1.10 \t\t 1.10 \t\t 1.10 \t\t 1.10\n";
  ofs << "bgnorm   lnN    1.00 \t\t 1.30 \t\t 1.30 \t\t 1.30\n";
  //alpha  shapeN2    -           1   uncertainty on background shape and normalization
  //sigma  shapeN2    0.5         -   uncertainty on signal resolution. Assume the histogram is a 2 sigma shift, 
  //                              so divide the unit gaussian by 2 before doing the interpolation
  ofs.close();
  
  TFile* fout = new TFile("dataCardShapes_NoWZ.root", "RECREATE");
  wwz_ZZ->h_disc_ZZ->Write("WWZ_bkg_ZZ");
  wwz_ttZ->h_disc_ZZ->Write("WWZ_bkg_ttZ");
  wwz_signal->h_disc_ZZ->Write("WWZ_signal");
  wwz_WZ->h_disc_ZZ->Write("WWZ_bkg_WZ");
  h_data->Write("data_obs");
  fout->Close();

  
  
  return 0;
}
