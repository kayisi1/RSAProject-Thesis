#supp_musicRSA.py
'''

code to perform RSA analysis (Kyle Aiysi)

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

#from sklearn.linear_model import LogisticRegression
paramsDict = fp.fmri_params()

#set root path
root_path = paramsDict['root_path']
if paramsDict['debugMode']:
	print('using root path...')
	print(root_path)

#path to first level directory
firstlevelDir = os.path.join(root_path,paramsDict['fmriBidsPath'],paramsDict['firstlevelDir'])

'''
if paramsDict['debugMode']:
	print('Running Representational Similarity Analysis\n'+'saving results to: \n'+os.path.join(firstlevelDir,paramsDict['RSAgroupDir'],paramsDict['preprocTag'] + '*.csv'))
	#ask about path^
	#saving on group level or subject level?
'''
"""
ANALYSIS
"""

RSAData = pd.DataFrame(data=None, index=None, columns=['subject', 'ROI', 'nvox', 'ED1', 'LD1', 'ED2', 'LD2', 'D1_CS', 'D2_CS'])
RDMTtestData = pd.DataFrame(data=None, index=None, columns=['subject', 'ROI', 'nvox', 'D1on_diag_avg', 'D1off_diag_avg', 'D1on_minus_off', 'D2on_diag_avg', 'D2off_diag_avg', 'D2on_minus_off', 'pairedTTestscore'])

#loop through subjects
try:
	subject_ids = [paramsDict['slurm2SubjectIDs'][int(sys.argv[1])]]
	print('runnin!'+subject_ids[0])
except:
	subject_ids = paramsDict['SubjectIDs']
	print('no sys input using all subs!')

#grab the info about how we are getting voxels/ROIs
roiKeyName,useTTflag,mf_flag,null1,null2 = myfuncs.loadparamtags(paramsDict)
isub_ROI_scores = []
for isub in subject_ids: #paramsDict['SubjectIDs']: [subject_id]
	
	## init placeholder for this outdat (subXroiXscores)
	OutData = pd.DataFrame(data = None, index = None ,columns=['subject','roi_group','pctTTvox','nVoxUsed'])

	## load the gray matter mask for this sub
	try:
		#maskFile = glob.glob(os.path.join(firstlevelDir,paramsDict['cleanfmriDir'].replace('???',isub.replace('par','')),isub+'_'+'T1W_dsegbrain'+'.nii.gz')) #parB355Schaefer40017.nii.gz
		maskFile = glob.glob(os.path.join(firstlevelDir,paramsDict['cleanfmriDir'].replace('???',isub.replace('par','')),isub+'Schaefer40017dwn'+'.nii.gz')) #parB355Schaefer40017.nii.gz
		mask_data = nib.load(maskFile[0])
	except:
		if paramsDict['debugMode']:
			print('couldnt find gray fmri mask for '+isub)

	## grab the loop ids for the 6 heard in all expo and rexpo runs
	eventFile = glob.glob(os.path.join(firstlevelDir,paramsDict['eventFilesPathLoop'].replace('???',isub.replace('par','')),paramsDict['EventNames']['expoFLL']['run-1'],'*.txt'))
	runEvent_dct = myfuncs.createEventTimingStimDct(paramsDict,eventFile,paramsDict['nLoopReps']) 
	loops2include = [*runEvent_dct.keys()]


	currROI_corr_scores = []
	## loop through each ROI we want to examine seperately 
	for iroigroup in paramsDict[roiKeyName].keys(): #paramsDict['FSiROINames2idx'].keys()
		if paramsDict['debugMode']:
			print('\n'+'working on RSA matrices for ROI: '+iroigroup)


		currROI = myfuncs.singleMaskFromMultipleV2(root_path,roiKeyName,iroigroup,mask_data,paramsDict)

		#currmask,currdseg = myfuncs.createFSbrainMask(segFile,paramsDict)

		## for this sub, gather all the data and labels for the SVM (also for specific ROI)
		#use mask based on union of all fold TT masks parN967_pre4_clndbold_perception_Mega_TTloop1p00GLM_RMSE_Nsigfolds.nii.gz
		if paramsDict['useTTGLM_maps']:
			#use mask based on union of all fold TT masks parN967_pre4_clndbold_perception_Mega_TTloop1p00GLM_RMSE_Nsigfolds.nii.gz
			ttmsk = os.path.join(firstlevelDir,paramsDict['GLMMF_rslts_fmriDir'].replace('???',isub.replace('par','')),isub+'_'+paramsDict['preprocTag']+'_clndbold_perception_Mega_TT'+paramsDict['TT_mode']+[*paramsDict['TT_window'].keys()][0]+'GLM_RMSE_Nsigfolds'+'.nii.gz')
		elif paramsDict['useONGLM_maps']:
			ttmsk = os.path.join(firstlevelDir,paramsDict['GLMMF_rslts_fmriDir'].replace('???',isub.replace('par','')),isub+'_'+paramsDict['preprocTag']+'_clndbold_perception_Mega_ON'+paramsDict['ON_mode']+'GLM_TVAL_Nsigfolds'+'.nii.gz')
		svmdata, pctTTvox, nvoxused = myfuncs.collectData4classification(currROI,firstlevelDir,isub,paramsDict,loops2include,mf_flag,ttmsk)
		

		#this path good to save out pictures, RDMs, etc
		#corrmatname.to_string(suboutfile), make into dataframe, to_csv, 'saving x to y online'
		#could also choose to save out on diagonal-off diagonal matrix in this loop
		suboutfile1 = os.path.join(firstlevelDir,paramsDict['RSAROI_indroi_fmriDir'].replace('???',isub.replace('par','')),paramsDict['preprocTag']+useTTflag+'_'+roiKeyName[2]+iroigroup+'_D1_corrmat.csv')
		suboutfile2 = os.path.join(firstlevelDir,paramsDict['RSAROI_indroi_fmriDir'].replace('???',isub.replace('par','')),paramsDict['preprocTag']+useTTflag+'_'+roiKeyName[2]+iroigroup+'_D2_corrmat.csv')
		
		OutData = OutData.append({'subject':isub,'roi_group':iroigroup,'pctTTvox':pctTTvox,'nVoxUsed':nvoxused}, ignore_index=True)
		

		if nvoxused >= paramsDict['minNVox4RSA']: #pctTTvox > 0:
			svmdatadf = pd.DataFrame(svmdata)
		
			d1earlystimulus_averages = [] #6 after gone through each stimulus
			d1latestimulus_averages = [] #6 after gone through each stimulus
			d2earlystimulus_averages = [] #6 after gone through each stimulus
			d2latestimulus_averages = [] #6 after gone through each stimulus

			for uniqueloop in np.unique(svmdatadf['label']):
				#splits original dataframe into rows from a given stimulus
				loop_rows = svmdatadf.loc[np.where(svmdatadf['label'] == uniqueloop)]
				#print(loop_rows)

				#filter days into halves
				d1_early = loop_rows.iloc[np.where(loop_rows['task'] == 'expoSTG')]['X']
				d1_late = loop_rows.iloc[np.where(loop_rows['task'] == 'expoFLL')]['X']
				d2_early = loop_rows.iloc[np.where((loop_rows['task'] == 'reexpoFLL') & ((loop_rows['run'] == 'run-1') | (loop_rows['run'] == 'run-2')))]['X']
				d2_late = loop_rows.iloc[np.where((loop_rows['task'] == 'reexpoFLL') & ((loop_rows['run'] == 'run-3') | (loop_rows['run'] == 'run-4')))]['X']
				
				#convert into dataframe
				d1_early_df = pd.DataFrame(np.row_stack(d1_early))
				d1_late_df = pd.DataFrame(np.row_stack(d1_late))
				d2_early_df = pd.DataFrame(np.row_stack(d2_early))
				d2_late_df = pd.DataFrame(np.row_stack(d2_late))

				#find average of columns
				avg_d1_early = d1_early_df.mean()
				avg_d1_late = d1_late_df.mean()
				avg_d2_early = d2_early_df.mean()
				avg_d2_late = d2_late_df.mean()

				#append to larger array of averages
				d1earlystimulus_averages.append(avg_d1_early)
				d1latestimulus_averages.append(avg_d1_late)
				d2earlystimulus_averages.append(avg_d2_early)
				d2latestimulus_averages.append(avg_d2_late)
				#figure out how to save out these results specific to subject and stimulus
					#this is for just 1 subject, 1 roi, 1 stimulus (one of the voxel x 8s in diagram)

			#do correlation measure on this level, save to array on level before in loop outside of this(ROI level)
				#pdb.set_trace()
			
			#concatenate dataframes together
			d1early_stimdf = pd.concat(d1earlystimulus_averages, axis=1)
			d1late_stimdf = pd.concat(d1latestimulus_averages, axis=1)
			d2early_stimdf = pd.concat(d2earlystimulus_averages, axis=1)
			d2late_stimdf = pd.concat(d2latestimulus_averages, axis=1)

			#Day 1 Correlation Matrix
			corr_mat_d1 = d1early_stimdf.apply(lambda x: d1late_stimdf.corrwith(x, axis=0, method='spearman')) #pearson
			corr_mat_d1.columns = ['S866_early', 'S869_early', 'S874_early', 'S886_early', 'S939_early', 'S964_early']
			corr_mat_d1.index = ['S866_late', 'S869_late', 'S874_late', 'S886_late', 'S939_late', 'S964_late']

			#Day 2 Correlation Matrix
			corr_mat_d2 = d2early_stimdf.apply(lambda x: d2late_stimdf.corrwith(x, axis=0, method='spearman'))
			corr_mat_d2.columns = ['S866_early', 'S869_early', 'S874_early', 'S886_early', 'S939_early', 'S964_early']
			corr_mat_d2.index = ['S866_late', 'S869_late', 'S874_late', 'S886_late', 'S939_late', 'S964_late']
			

			#Save correlation matrices to csv
			corr_mat_d1.to_csv(suboutfile1)
			corr_mat_d2.to_csv(suboutfile2)

		else:
			corr_mat_d1 = pd.DataFrame(data=None,index=['S866_late', 'S869_late', 'S874_late', 'S886_late', 'S939_late', 'S964_late'],columns=['S866_early', 'S869_early', 'S874_early', 'S886_early', 'S939_early', 'S964_early'])
			corr_mat_d2 = pd.DataFrame(data=None,index=['S866_late', 'S869_late', 'S874_late', 'S886_late', 'S939_late', 'S964_late'],columns=['S866_early', 'S869_early', 'S874_early', 'S886_early', 'S939_early', 'S964_early'])
			#Save correlation matrices to csv
			corr_mat_d1.to_csv(suboutfile1)
			corr_mat_d2.to_csv(suboutfile2)

	print('done with ' +isub + '!')
	OutData.to_csv(os.path.join(firstlevelDir,paramsDict['RSAROI_fmriDir'].replace('???',isub.replace('par','')),paramsDict['preprocTag']+useTTflag+'_'+roiKeyName[2]+'_nVox.csv'))		
	#np.diag, np.tril, np.triu for diagonal and off diag
			
	#another for loop, goes through ROIs, reads in correlation matrices, do on-off diag, and T-tests (in another script)
