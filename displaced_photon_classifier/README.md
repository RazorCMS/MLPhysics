This directory contains Python scripts to train and evaluate XGBoost models to discriminate signal from background in the displaced photon SUSY search.

- `PhotonTraining.py`: loads the data from ROOT, trains the model, creates a ROC curve, and saves the model to a pickle file.
- `PhotonTraining_onQCD.py`, `PhotonTraining_onGJets.py`: load the model saved by `PhotonTraining.py` and test on the other background processes.
- `PhotonTraining_onGJets_rootupdated.py` is similar to the python files listed above: it loads the model saved by `PhotonTraining.py` (currently set to run on GMSB vs. QCD) and tests on another background (currently set to photons+jets). This outputs ROC curves and also a ROOT file of the input file with the discriminator values appended.
- `PhotonTraining_overlay.py`, `PhotonTraining_overlay_lifetimes.py`, `PhotonTraining_VariableOverlay.py`, `PhotonTraining_Variable7_timing.py` are similar files to PhotonTraining.py but they take multiple signal files and make overlayed ROC curves to compare lifetimes, masses, or the inclusion of different variables

Input files are located on lxplus:
`/afs/cern.ch/work/g/gkopp/Thesis/MC_Nov24_noIsoCut_barrelskim/`
and duplicated to public area
`/afs/cern.ch/work/g/gkopp/public/ThesisROOTfiles`

### Setup
```
git clone https://github.com/RazorCMS/MLPhysics.git
cd MLPhysics/displaced_photon_classifier/
cmsrel CMSSW_9_3_3
cd CMSSW_9_3_3/src/
cmsenv
cd -
python PhotonTraining.py
```
This is currently set to run on the files from `/afs/cern.ch/work/g/gkopp/public/ThesisROOTfiles/`. These files are from the Monte Carlo that Zhicai made, and I skimmed the ROOT files on pho1Pt > 0 and for in barrel photons.
*Note the discriminator values are set to -1 for the events that have been used in the training set* - and these events should never be used again since they would make it biased.

This will create the initial BDT and produce ROOT files for the QCD background and the GMSB signal.  To run the BDT on a different input ROOT file and save the score for each event, use the `EvalBDT.py` script.  This takes as arguments the path to the input ROOT file, the chunk size (number of events to process in one job), and a number indicating which chunk to process.  Example:
```
python EvalBDT.py --chunk-size 5000 --chunk 2 /eos/cms/store/group/phys_susy/razor/Run2Analysis/DelayedPhotonAnalysis/2016/V4p1_private_REMINIAOD/withcut/GMSB_L150TeV_Ctau20000cm_13TeV-pythia8.root
```
WARNING: do not run `EvalBDT.py` on any ROOT files containing events used in the original BDT training.  Reusing the training events will give a biased distribution of BDT discriminator values.
