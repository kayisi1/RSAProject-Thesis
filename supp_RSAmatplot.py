#supp_RSAmatplot
'''

code to plot matrices (Kyle)

Inputs:
Outputs:

'''

import musicpie_fmri_params as fp
import musicpie_fmri_funcs as myfuncs

import pdb
import os
import glob
import numpy as np
import pandas as pd
import re
import sys
import pickle

import nibabel as nib
import seaborn as sns
#from nilearn import image as nil
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
#from sklearn.decomposition import PCA
from scipy.stats import norm,zscore,pearsonr,stats
from scipy import spatial
from sklearn.model_selection import PredefinedSplit


#from sklearn.svm import SVC
#from sklearn.model_selection import cross_val_score,cross_validate,cross_val_predict
#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix#,precision_score,recall_score #classification_report

import matplotlib.pyplot as plt

roisubpath = 'roisub_results'
roigrouppath = 'roigroup_results'
roiplotpath = 'roiplot_results'

paramsDict = fp.fmri_params()

#set root path
root_path = paramsDict['root_path']
if paramsDict['debugMode']:
	print('using root path...')
	print(root_path)

#path to first level directory
firstlevelDir = os.path.join(root_path,paramsDict['fmriBidsPath'],paramsDict['firstlevelDir'])

try:
	subject_ids = [paramsDict['slurm2SubjectIDs'][int(sys.argv[1])]]
	print('runnin!'+subject_ids[0])
except:
	subject_ids = paramsDict['SubjectIDs']
	print('no sys input using all subs!')

roiKeyName,useTTflag,mf_flag,null1,null2 = myfuncs.loadparamtags(paramsDict)

randomisubs = ['parN967','parI882','parY572','parS709', 'parX456']

somerois = ["LH_SomMotB_Aud_1","RH_SomMotB_Aud_1","LH_SomMotB_Aud_2","RH_SomMotB_Aud_2","LH_SomMotB_Aud_3","RH_SomMotB_Aud_3","LH_SomMotB_Aud_4",
"LH_SomMotB_Ins_1","RH_SomMotB_Ins_1","LH_SomMotB_S2_1","RH_SomMotB_S2_1","LH_SomMotB_S2_2","RH_SomMotB_S2_2","LH_SomMotB_S2_3","RH_SomMotB_S2_3",
"LH_SomMotB_S2_4","RH_SomMotB_S2_4","LH_SomMotB_S2_5","RH_SomMotB_S2_5","LH_SomMotB_S2_6","RH_SomMotB_S2_6","RH_SomMotB_S2_7","RH_SomMotB_S2_8",
"LH_SomMotB_Cent_1","RH_SomMotB_Cent_1","LH_SomMotB_Cent_2","RH_SomMotB_Cent_2","LH_SomMotB_Cent_3","RH_SomMotB_Cent_3","LH_SomMotB_Cent_4",
"LH_SomMotB_Cent_5","LH_TempPar_1","RH_TempPar_1","LH_TempPar_2","RH_TempPar_2","LH_TempPar_3","RH_TempPar_3","LH_TempPar_4","RH_TempPar_4",
"LH_TempPar_5","RH_TempPar_5","LH_TempPar_6","RH_TempPar_6","RH_TempPar_7","RH_TempPar_8","RH_TempPar_9","RH_TempPar_10"]

for isub in randomisubs:
	for iroigroup in somerois:
		d1path = os.path.join(firstlevelDir,paramsDict['CLROI_indroi_fmriDir'].replace('???',isub.replace('par','')),paramsDict['preprocTag']+useTTflag+'_'+roiKeyName[2]+iroigroup+'_D1_corrmat.csv')
		d2path = os.path.join(firstlevelDir,paramsDict['CLROI_indroi_fmriDir'].replace('???',isub.replace('par','')),paramsDict['preprocTag']+useTTflag+'_'+roiKeyName[2]+iroigroup+'_D2_corrmat.csv')
		
		d1mat = pd.read_csv(d1path)
		d2mat = pd.read_csv(d2path)
		d1mat = d1mat.drop(columns=['Unnamed: 0'])
		d2mat = d2mat.drop(columns=['Unnamed: 0'])

		rownames = {0:'S866_late', 1: 'S869_late', 2: 'S874_late', 3:'S886_late',4:'S939_late',5:'S964_late'}
		d1mat = d1mat.rename(index = rownames)
		d2mat = d2mat.rename(index = rownames)

		suboutfile1 = open(os.path.join(firstlevelDir,paramsDict['CLROI_indroi_fmriDir'].replace('???',isub.replace('par','')),paramsDict['preprocTag']+useTTflag+'_'+roiKeyName[2]+iroigroup+'_D1_corrmatplot.pdf'),"wb+") 
		suboutfile2 = open(os.path.join(firstlevelDir,paramsDict['CLROI_indroi_fmriDir'].replace('???',isub.replace('par','')),paramsDict['preprocTag']+useTTflag+'_'+roiKeyName[2]+iroigroup+'_D2_corrmatplot.pdf'),"wb+")

		fig, ax = plt.subplots(figsize=(12,12))
		sns.heatmap(d1mat, cmap='viridis', annot=True)
		plt.xlabel('Stimulus')
		plt.ylabel('Stimulus')
		plt.title(isub + 'Day 1 Stimulus Correlation Matrix (' + iroigroup + ')')
		
		plt.savefig(suboutfile1, format='pdf')

		fig, ax = plt.subplots(figsize=(12,12))
		sns.heatmap(d2mat, cmap='viridis', annot=True)
		plt.xlabel('Stimulus')
		plt.ylabel('Stimulus')
		plt.title(isub + 'Day 2 Stimulus Correlation Matrix (' + iroigroup + ')')
		
		plt.savefig(suboutfile2, format='pdf')

		plt.close('all')
		