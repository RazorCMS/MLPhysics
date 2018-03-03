This directory contains Python scripts to train and evaluate XGBoost models to discriminate signal from background in the displaced photon SUSY search.

- `PhotonTraining.py`: loads the data from ROOT, trains the model, creates a ROC curve, and saves the model to a pickle file.
- `PhotonTraining_onQCD.py`, `PhotonTraining_onGJets.py`: load the model saved by `PhotonTraining.py` and test on the other background processes.
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
