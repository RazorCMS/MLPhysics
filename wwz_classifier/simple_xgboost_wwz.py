#!/usr/bin/env python

import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve
import root_pandas as rp
import numpy as np

##Getting ROOT files into pandas
df_signal = rp.read_root('/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_WWZJetsTo4L2Nu_4f_TuneCUETP8M1_13TeV_aMCatNLOFxFx_pythia8.root', 'WWZAnalysis', columns=['MET','lep1Pt','lep2Pt'])
df_bkg = rp.read_root('/Users/cmorgoth/Work/data/WWZanalysis/MC/WWZAnalysis_ZZTo4L_13TeV-amcatnloFXFX-pythia8.root', 'WWZAnalysis', columns=['MET','lep1Pt','lep2Pt'])


# print df_signal.head()
# print df_signal.tail()

# print df_bkg.head()
# print df_bkg.tail()

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
model = XGBClassifier()
model.fit(x_train, y_train)

print( dir(model) )

# make predictions for test data
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#get roc curve
#roc = roc_curve(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred)

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
