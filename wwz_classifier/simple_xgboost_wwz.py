#!/usr/bin/env python

import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve
import root_pandas as rp
import numpy as np


##Define variables to be used
variables = ['MET','METPhi','lep1Pt','lep2Pt','lep3Pt','lep4Pt','NJet20','NJet30','NBJet20','NBJet30','lep1Phi','lep2Phi','lep3Phi','lep4Phi','lep1Eta','lep2Eta','lep3Eta','lep4Eta','ZMass','ZPt','lep3MT','lep4MT','lep34MT','phi0','theta0','phi','theta1','theta2','phiH','minDRJetToLep3','minDRJetToLep4'] 
##Getting ROOT files into pandas
#df_signal = rp.read_root('/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_WWZJetsTo4L2Nu_4f_TuneCUETP8M1_13TeV_aMCatNLOFxFx_pythia8.root', 'WWZAnalysis', columns=['MET','lep1Pt','lep2Pt','lep3Pt','lep4Pt'])
#df_bkg = rp.read_root('/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_ZZTo4L_13TeV-amcatnloFXFX-pythia8.root', 'WWZAnalysis', columns=['MET','lep1Pt','lep2Pt','lep3Pt','lep4Pt'])
df_signal = rp.read_root('/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_WWZJetsTo4L2Nu_4f_TuneCUETP8M1_13TeV_aMCatNLOFxFx_pythia8.root',
                             'WWZAnalysis', columns=variables)
df_bkg = rp.read_root('/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_ZZTo4L_13TeV-amcatnloFXFX-pythia8.root',
                          'WWZAnalysis', columns=variables)


##Getting Sample weights
weights_signal = rp.read_root('/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_WWZJetsTo4L2Nu_4f_TuneCUETP8M1_13TeV_aMCatNLOFxFx_pythia8.root',
                                  'WWZAnalysis', columns=['weight'])
weights_bkg = rp.read_root('/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_ZZTo4L_13TeV-amcatnloFXFX-pythia8.root',
                          'WWZAnalysis', columns=['weight'])


## We care about the sign of the sample only, we don't care about the xsec weight
#weights_sign_signal = np.sign(weights_signal)
#weights_sign_bkg = np.sign(weights_bkg)
weights_sign_signal = weights_signal
weights_sign_bkg = weights_bkg

#print weights_sign_signal.head()
#print weights_sign_bkg.head()

##Getting a numpy array out of two pandas data frame
sample_weights = np.concatenate([weights_sign_bkg.values, weights_sign_signal.values])
print sample_weights

#getting a numpy array from two pandas data frames
x = np.concatenate([df_bkg.values,df_signal.values])
#creating numpy array for target variables
y = np.concatenate([np.zeros(len(df_bkg)),
                        np.ones(len(df_signal))])

# split data into train and test sets
seed = 7
test_size = 0.2
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBClassifier(max_depth=8, gamma=1, silent=True)
model.fit(x_train, y_train, sample_weight=sample_weights)

#print( dir(model) )
print model

# make predictions for test data
#y_pred = model.predict(x_test)
y_pred = model.predict_proba(x_test)[:, 1]
predictions = [round(value) for value in y_pred]

print y_pred

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#get roc curve
#roc = roc_curve(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred)

print fpr
print tpr
#plot roc curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
plt.savefig('myroc.png')


# plot
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.show()
plt.savefig('myImportances.png')
