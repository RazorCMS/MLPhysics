//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Sun Aug 27 08:49:10 2017 by ROOT version 6.10/02
// from TTree WWZAnalysis/Info on selected razor inclusive events
// found on file: WWZAnalysis_WWZJetsTo4L2Nu_4f_TuneCUETP8M1_13TeV_aMCatNLOFxFx_pythia8.root
//////////////////////////////////////////////////////////

#ifndef WWZAnalysis_h
#define WWZAnalysis_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.

class WWZAnalysis {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   Float_t         weight;
   Float_t         pileupWeight;
   Float_t         pileupWeightUp;
   Float_t         pileupWeightDown;
   Float_t         triggerEffWeight;
   Float_t         triggerEffSFWeight;
   UInt_t          run;
   UInt_t          lumi;
   UInt_t          event;
   UInt_t          NPU;
   UInt_t          nPV;
   Float_t         MET;
   Float_t         MET_JESUp;
   Float_t         MET_JESDown;
   Float_t         METPhi;
   Int_t           NJet20;
   Int_t           NJet30;
   Int_t           NBJet20;
   Int_t           NBJet30;
   Int_t           lep1Id;
   Float_t         lep1Pt;
   Float_t         lep1Eta;
   Float_t         lep1Phi;
   Int_t           lep2Id;
   Float_t         lep2Pt;
   Float_t         lep2Eta;
   Float_t         lep2Phi;
   Int_t           lep3Id;
   Float_t         lep3Pt;
   Float_t         lep3Eta;
   Float_t         lep3Phi;
   Int_t           lep4Id;
   Float_t         lep4Pt;
   Float_t         lep4Eta;
   Float_t         lep4Phi;
   Float_t         ZMass;
   Float_t         ZPt;
   Float_t         lep3MT;
   Float_t         lep4MT;
   Float_t         lep34MT;
   Float_t         phi0;
   Float_t         theta0;
   Float_t         phi;
   Float_t         theta1;
   Float_t         theta2;
   Float_t         phiH;
   Float_t         minDRJetToLep3;
   Float_t         minDRJetToLep4;
   Bool_t          HLTDecision[300];

   // List of branches
   TBranch        *b_weight;   //!
   TBranch        *b_pileupWeight;   //!
   TBranch        *b_pileupWeightUp;   //!
   TBranch        *b_pileupWeightDown;   //!
   TBranch        *b_triggerEffWeight;   //!
   TBranch        *b_triggerEffSFWeight;   //!
   TBranch        *b_run;   //!
   TBranch        *b_lumi;   //!
   TBranch        *b_event;   //!
   TBranch        *b_npu;   //!
   TBranch        *b_nPV;   //!
   TBranch        *b_MET;   //!
   TBranch        *b_MET_JESUp;   //!
   TBranch        *b_MET_JESDown;   //!
   TBranch        *b_METPhi;   //!
   TBranch        *b_NJet20;   //!
   TBranch        *b_NJet30;   //!
   TBranch        *b_NBJet20;   //!
   TBranch        *b_NBJet30;   //!
   TBranch        *b_lep1Id;   //!
   TBranch        *b_lep1Pt;   //!
   TBranch        *b_lep1Eta;   //!
   TBranch        *b_lep1Phi;   //!
   TBranch        *b_lep2Id;   //!
   TBranch        *b_lep2Pt;   //!
   TBranch        *b_lep2Eta;   //!
   TBranch        *b_lep2Phi;   //!
   TBranch        *b_lep3Id;   //!
   TBranch        *b_lep3Pt;   //!
   TBranch        *b_lep3Eta;   //!
   TBranch        *b_lep3Phi;   //!
   TBranch        *b_lep4Id;   //!
   TBranch        *b_lep4Pt;   //!
   TBranch        *b_lep4Eta;   //!
   TBranch        *b_lep4Phi;   //!
   TBranch        *b_ZMass;   //!
   TBranch        *b_ZPt;   //!
   TBranch        *b_lep3MT;   //!
   TBranch        *b_lep4MT;   //!
   TBranch        *b_lep34MT;   //!
   TBranch        *b_phi0;   //!
   TBranch        *b_theta0;   //!
   TBranch        *b_phi;   //!
   TBranch        *b_theta1;   //!
   TBranch        *b_theta2;   //!
   TBranch        *b_phiH;   //!
   TBranch        *b_minDRJetToLep3;   //!
   TBranch        *b_minDRJetToLep4;   //!
   TBranch        *b_HLTDecision;   //!

   WWZAnalysis(TTree *tree=0);
   virtual ~WWZAnalysis();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef WWZAnalysis_cxx
WWZAnalysis::WWZAnalysis(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("WWZAnalysis_WWZJetsTo4L2Nu_4f_TuneCUETP8M1_13TeV_aMCatNLOFxFx_pythia8.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("WWZAnalysis_WWZJetsTo4L2Nu_4f_TuneCUETP8M1_13TeV_aMCatNLOFxFx_pythia8.root");
      }
      f->GetObject("WWZAnalysis",tree);

   }
   Init(tree);
}

WWZAnalysis::~WWZAnalysis()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t WWZAnalysis::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t WWZAnalysis::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void WWZAnalysis::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("weight", &weight, &b_weight);
   fChain->SetBranchAddress("pileupWeight", &pileupWeight, &b_pileupWeight);
   fChain->SetBranchAddress("pileupWeightUp", &pileupWeightUp, &b_pileupWeightUp);
   fChain->SetBranchAddress("pileupWeightDown", &pileupWeightDown, &b_pileupWeightDown);
   fChain->SetBranchAddress("triggerEffWeight", &triggerEffWeight, &b_triggerEffWeight);
   fChain->SetBranchAddress("triggerEffSFWeight", &triggerEffSFWeight, &b_triggerEffSFWeight);
   fChain->SetBranchAddress("run", &run, &b_run);
   fChain->SetBranchAddress("lumi", &lumi, &b_lumi);
   fChain->SetBranchAddress("event", &event, &b_event);
   fChain->SetBranchAddress("NPU", &NPU, &b_npu);
   fChain->SetBranchAddress("nPV", &nPV, &b_nPV);
   fChain->SetBranchAddress("MET", &MET, &b_MET);
   fChain->SetBranchAddress("MET_JESUp", &MET_JESUp, &b_MET_JESUp);
   fChain->SetBranchAddress("MET_JESDown", &MET_JESDown, &b_MET_JESDown);
   fChain->SetBranchAddress("METPhi", &METPhi, &b_METPhi);
   fChain->SetBranchAddress("NJet20", &NJet20, &b_NJet20);
   fChain->SetBranchAddress("NJet30", &NJet30, &b_NJet30);
   fChain->SetBranchAddress("NBJet20", &NBJet20, &b_NBJet20);
   fChain->SetBranchAddress("NBJet30", &NBJet30, &b_NBJet30);
   fChain->SetBranchAddress("lep1Id", &lep1Id, &b_lep1Id);
   fChain->SetBranchAddress("lep1Pt", &lep1Pt, &b_lep1Pt);
   fChain->SetBranchAddress("lep1Eta", &lep1Eta, &b_lep1Eta);
   fChain->SetBranchAddress("lep1Phi", &lep1Phi, &b_lep1Phi);
   fChain->SetBranchAddress("lep2Id", &lep2Id, &b_lep2Id);
   fChain->SetBranchAddress("lep2Pt", &lep2Pt, &b_lep2Pt);
   fChain->SetBranchAddress("lep2Eta", &lep2Eta, &b_lep2Eta);
   fChain->SetBranchAddress("lep2Phi", &lep2Phi, &b_lep2Phi);
   fChain->SetBranchAddress("lep3Id", &lep3Id, &b_lep3Id);
   fChain->SetBranchAddress("lep3Pt", &lep3Pt, &b_lep3Pt);
   fChain->SetBranchAddress("lep3Eta", &lep3Eta, &b_lep3Eta);
   fChain->SetBranchAddress("lep3Phi", &lep3Phi, &b_lep3Phi);
   fChain->SetBranchAddress("lep4Id", &lep4Id, &b_lep4Id);
   fChain->SetBranchAddress("lep4Pt", &lep4Pt, &b_lep4Pt);
   fChain->SetBranchAddress("lep4Eta", &lep4Eta, &b_lep4Eta);
   fChain->SetBranchAddress("lep4Phi", &lep4Phi, &b_lep4Phi);
   fChain->SetBranchAddress("ZMass", &ZMass, &b_ZMass);
   fChain->SetBranchAddress("ZPt", &ZPt, &b_ZPt);
   fChain->SetBranchAddress("lep3MT", &lep3MT, &b_lep3MT);
   fChain->SetBranchAddress("lep4MT", &lep4MT, &b_lep4MT);
   fChain->SetBranchAddress("lep34MT", &lep34MT, &b_lep34MT);
   fChain->SetBranchAddress("phi0", &phi0, &b_phi0);
   fChain->SetBranchAddress("theta0", &theta0, &b_theta0);
   fChain->SetBranchAddress("phi", &phi, &b_phi);
   fChain->SetBranchAddress("theta1", &theta1, &b_theta1);
   fChain->SetBranchAddress("theta2", &theta2, &b_theta2);
   fChain->SetBranchAddress("phiH", &phiH, &b_phiH);
   fChain->SetBranchAddress("minDRJetToLep3", &minDRJetToLep3, &b_minDRJetToLep3);
   fChain->SetBranchAddress("minDRJetToLep4", &minDRJetToLep4, &b_minDRJetToLep4);
   fChain->SetBranchAddress("HLTDecision", HLTDecision, &b_HLTDecision);
   Notify();
}

Bool_t WWZAnalysis::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void WWZAnalysis::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t WWZAnalysis::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef WWZAnalysis_cxx
