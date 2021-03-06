#include <iostream>
#include <fstream>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TGraph.h>
#include <TDirectory.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TString.h>
#include <THStack.h>

int main()
{

  TCanvas* c = new TCanvas( "c", "c", 800, 600 );
  c->SetHighLightColor(2);
  c->SetFillColor(0);
  c->SetBorderMode(0);
  c->SetBorderSize(2);
  c->SetFrameBorderMode(0);
  c->SetFrameBorderMode(0);

  const int npoints = 50;
  
  TFile* fin = new TFile("/Users/cmorgoth/Work/data/WWZanalysis/MC/NewLeptonID/WWZAnalysis_WWZJetsTo4L2Nu_4f_TuneCUETP8M1_13TeV_aMCatNLOFxFx_pythia8_1pb_weighted_lep34Mass_ZZDiscriminator_ttZDiscriminator.root", "READ");
  TTree* tree_signal = (TTree*)fin->Get("WWZAnalysis");
  
  //For Bkg ZZ
  fin = new TFile("/Users/cmorgoth/Work/data/WWZanalysis/MC/NewLeptonID/WWZAnalysis_ZZTo4L_13TeV_powheg_pythia8_1pb_weighted_lep34Mass_ZZDiscriminator_ttZDiscriminator.root", "READ");
  TTree* tree_ZZ = (TTree*)fin->Get("WWZAnalysis");

  //For Bkg ttZ
  fin = new TFile("/Users/cmorgoth/Work/data/WWZanalysis/MC/NewLeptonID/WWZAnalysis_ttZJets_13TeV_madgraphMLM_1pb_weighted_lep34Mass_ZZDiscriminator_ttZDiscriminator.root", "READ");
  TTree* tree_ttZ = (TTree*)fin->Get("WWZAnalysis");

  //For Bkg WZ
  fin = new TFile("/Users/cmorgoth/Work/data/WWZanalysis/MC/NewLeptonID/WWZAnalysis_WZTo3LNu_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8_1pb_weighted_lep34Mass_ZZDiscriminator_ttZDiscriminator.root", "READ");
  TTree* tree_WZ = (TTree*)fin->Get("WWZAnalysis");

  //Singal
  tree_signal->Draw(Form("disc_ZZ>>tmp_signal(%d,0,1)",npoints), "", "goff");
  TH1F* h_signal = (TH1F*)gDirectory->Get("tmp_signal");
  h_signal->Scale(1.0/h_signal->Integral());
  h_signal->SetLineColor(kBlue);
  h_signal->SetLineWidth(2);
  //ZZ
  tree_ZZ->Draw(Form("disc_ZZ>>tmp_ZZ(%d,0,1)", npoints), "", "goff");
  TH1F* h_ZZ = (TH1F*)gDirectory->Get("tmp_ZZ");
  h_ZZ->Scale(1.0/h_ZZ->Integral());
  h_ZZ->SetLineColor(kOrange+7);
  h_ZZ->SetLineWidth(2);

  //ttZ
  tree_ttZ->Draw(Form("disc_ZZ>>ttz_ZZ(%d,0,1)", npoints), "", "goff");
  TH1F* h_ttz_zz = (TH1F*)gDirectory->Get("ttz_ZZ");
  h_ttz_zz->Scale(1.0/h_ttz_zz->Integral());
  h_ttz_zz->SetLineColor(kMagenta-4);
  h_ttz_zz->SetLineWidth(2);

  //WZ
  tree_WZ->Draw(Form("disc_ZZ>>wz_ZZ(%d,0,1)", npoints), "", "goff");
  TH1F* h_wz_zz = (TH1F*)gDirectory->Get("wz_ZZ");
  h_wz_zz->Scale(1.0/h_wz_zz->Integral());
  h_wz_zz->SetLineColor(kGreen-3);
  h_wz_zz->SetLineWidth(2);
  
  h_signal->SetStats(0);
  h_signal->SetXTitle("disc_ZZ");
  h_signal->SetYTitle("Fraction of events");
  h_signal->SetTitle("");
  h_signal->Draw("HIST");
  h_ZZ->Draw("same+HIST");
  //h_ttz_zz->Draw("same+HIST");
  //h_wz_zz->Draw("same+HIST");

  
  
  TLegend* leg = new TLegend( 0.2, 0.76, 0.5, 0.89, NULL, "brNDC" );
  leg->SetBorderSize(0);
  leg->SetLineColor(1);
  leg->SetLineStyle(1);
  leg->SetLineWidth(1);
  leg->SetFillColor(0);
  leg->SetFillStyle(1001);
  leg->AddEntry( h_signal, "WWZ (4l)", "l" );
  leg->AddEntry( h_ZZ, "ZZ (4l)", "l" );
  //leg->AddEntry( h_ttz_zz, "ttZ (4l)", "l" );
  //leg->AddEntry( h_wz_zz, "WZ (4l)", "l" );
  leg->Draw();
  c->SetLogy();
  c->SaveAs("disc_ZZ_WWZ_vs_ZZ.pdf");

  THStack* stack = new THStack( "hs" , "Hgg Stack " );
  h_wz_zz->SetFillColor(kGreen-3);
  h_ttz_zz->SetFillColor(kMagenta-4);
  h_ZZ->SetFillColor(kOrange+7);
  stack->Add( h_ZZ, "histo" );
  stack->Add( h_ttz_zz, "histo" );
  stack->Add( h_wz_zz, "histo" );
  
  h_signal->Draw("");
  stack->Draw("same");
  h_signal->Draw("HIST+same");

  leg = new TLegend( 0.2, 0.76, 0.5, 0.89, NULL, "brNDC" );
  leg->SetBorderSize(0);
  leg->SetLineColor(1);
  leg->SetLineStyle(1);
  leg->SetLineWidth(1);
  leg->SetFillColor(0);
  leg->SetFillStyle(1001);
  leg->AddEntry( h_signal, "WWZ (4l)", "l" );
  leg->AddEntry( h_ZZ, "ZZ (4l)", "lf" );
  leg->AddEntry( h_ttz_zz, "ttZ (4l)", "lf" );
  leg->AddEntry( h_wz_zz, "WZ (4l)", "lf" );
  leg->Draw();
  c->SaveAs("disc_ZZ_stack.pdf");
  //ROC CURVE
  
  float bkg_eff[npoints];
  float signal_eff[npoints];
  for( int i = 1; i <= npoints; i++ )
    {
      bkg_eff[i-1]    = h_ZZ->Integral(i,npoints);
      signal_eff[i-1] = h_signal->Integral(i,npoints);
    }

  TGraph* roc_disc_zz = new TGraph(npoints, bkg_eff, signal_eff);
  roc_disc_zz->GetYaxis()->SetRangeUser(0.7,1.0);
  roc_disc_zz->GetYaxis()->SetTitle("Signal eff.");
  roc_disc_zz->GetXaxis()->SetTitle("Bkg. eff.");
  roc_disc_zz->SetTitle("");
  roc_disc_zz->SetLineColor(kBlue);
  roc_disc_zz->SetLineWidth(3);
  roc_disc_zz->Draw("AL");
  c->SetLogy(0);
  c->SaveAs("ROC_disc_ZZ_WWZ_vs_ZZ.pdf");
  
  //ttZ discriminator
  tree_signal->Draw(Form("disc_ttZ>>signal_ttz(%d,0,1)",npoints), "", "goff");
  TH1F* h_signal_ttz = (TH1F*)gDirectory->Get("signal_ttz");
  h_signal_ttz->Scale(1.0/h_signal_ttz->Integral());
  h_signal_ttz->SetLineColor(kBlue);
  h_signal_ttz->SetLineWidth(2);
  //ZZ
  tree_ttZ->Draw(Form("disc_ttZ>>ttZ_ttz(%d,0,1)", npoints), "", "goff");
  TH1F* h_ttZ_ttz = (TH1F*)gDirectory->Get("ttZ_ttz");
  h_ttZ_ttz->Scale(1.0/h_ttZ_ttz->Integral());
  h_ttZ_ttz->SetLineColor(kRed);
  h_ttZ_ttz->SetLineWidth(2);

  h_signal_ttz->SetStats(0);
  h_signal_ttz->SetXTitle("disc_ttZ");
  h_signal_ttz->SetYTitle("Fraction of events");
  h_signal_ttz->SetTitle("");
  h_signal_ttz->Draw("HIST");
  h_ttZ_ttz->Draw("same+HIST");


  TLegend* leg1 = new TLegend( 0.2, 0.76, 0.5, 0.89, NULL, "brNDC" );
  leg1->SetBorderSize(0);
  leg1->SetLineColor(1);
  leg1->SetLineStyle(1);
  leg1->SetLineWidth(1);
  leg1->SetFillColor(0);
  leg1->SetFillStyle(1001);
  leg1->AddEntry( h_signal_ttz, "WWZ (4l)", "l" );
  leg1->AddEntry( h_ttZ_ttz, "ttZ (4l)", "l" );
  leg1->Draw();
  c->SetLogy();
  c->SaveAs("disc_ttZ_WWZ_vs_ttZ.pdf");

  for( int i = 1; i <= npoints; i++ )
    {
      bkg_eff[i-1]    = h_ttZ_ttz->Integral(i,npoints);
      signal_eff[i-1] = h_signal_ttz->Integral(i,npoints);
    }

  TGraph* roc_disc_ttz = new TGraph(npoints, bkg_eff, signal_eff);
  roc_disc_ttz->GetYaxis()->SetRangeUser(0.0,1.0);
  roc_disc_ttz->GetYaxis()->SetTitle("Signal eff.");
  roc_disc_ttz->GetXaxis()->SetTitle("Bkg. eff.");
  roc_disc_ttz->SetTitle("");
  roc_disc_ttz->SetLineColor(kBlue);
  roc_disc_ttz->SetLineWidth(3);
  roc_disc_ttz->Draw("AL");
  c->SetLogy(0);
  c->SaveAs("ROC_disc_ttZ_WWZ_vs_ttZ.pdf");
  
 
  return 0;
}
