#supp_musicRSAttest
'''

code to perform RSA analysis (Kyle Aiysi)

Inputs:
Outputs:
only does t-test for ROIS with data from all subs (and each sub with at least 5 voxels)

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

def FDR_p(pvals):
    # Port of AFNI mri_fdrize.c
    assert np.all(pvals>=0) and np.all(pvals<=1)
    pvals[pvals < np.finfo(np.float_).eps] = np.finfo(np.float_).eps
    pvals[pvals == 1] = 1-np.finfo(np.float_).eps
    n = pvals.shape[0]

    qvals = np.zeros((n))
    sorted_ind = np.argsort(pvals)
    sorted_pvals = pvals[sorted_ind]
    qmin = 1.0
    for i in range(n-1,-1,-1):
        qval = (n * sorted_pvals[i])/(i+1)
        if qval > qmin:
            qval = qmin
        else:
            qmin = qval
        qvals[sorted_ind[i]] = qval

    # Estimate number of true positives m1 and adjust q
    if n >= 233:
        phist = np.histogram(pvals, bins=20, range=(0, 1))[0]
        sorted_phist = np.sort(phist[3:19])
        if np.sum(sorted_phist) >= 160:
            median4 = n - 20*np.dot(np.array([1, 2, 2, 1]), sorted_phist[6:10])/6
            median6 = n - 20*np.dot(np.array([1, 2, 2, 2, 2, 1]), sorted_phist[5:11])/10
            m1 = min(median4, median6)

            qfac = (n - m1)/n
            if qfac < 0.5:
                qfac = 0.25 + qfac**2
            qvals *= qfac

    return qvals

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
ANALYSIS
'''

#loop through subjects
#note this won't work as seperate jobs, need all sub info together.
subject_ids = paramsDict['SubjectIDs']
print('no sys input using all subs!')

#grab the info about how we are getting voxels/ROIs
roiKeyName,useTTflag,mf_flag,null1,null2 = myfuncs.loadparamtags(paramsDict)

#create initial outdata dataframe like classifier script FIX COLUMN NAMES
TTest_Data = pd.DataFrame(data=None, index=None, columns=['roi_group','nsubs', 'maxvox', 'minvox', 'avgvoxused', 'stdvox', 'd1_diffmax', 'd1_diffmin', 'd1_diffavg',
'd1_diffstd', 'd2_diffmax', 'd2_diffmin', 'd2_diffavg', 'd2_diffstd', 'tscore', 'pvalue'])

#init a placeholder brain in MNI
maskFile = glob.glob(os.path.join(root_path,paramsDict['schaeferMaskDir'],paramsDict['schaeferMNInii'])) #parB355Schaefer40017.nii.gz
mask_data = nib.load(maskFile[0])
dimsize = mask_data.header.get_zooms()

outtval_vol = np.zeros((mask_data.shape[0], mask_data.shape[1], mask_data.shape[2]))  # shape of the output
outsigmask_vol = np.zeros((mask_data.shape[0], mask_data.shape[1], mask_data.shape[2]))

for iroigroup in paramsDict[roiKeyName].keys(): #paramsDict['FSiROINames2idx'].keys()
	if paramsDict['debugMode']:
		print('\n'+'working on t-tests for ROI: '+iroigroup)

	#currROI = myfuncs.singleMaskFromMultipleV2(root_path,roiKeyName,iroigroup,mask_data,paramsDict)
	#currmask,currdseg = myfuncs.createFSbrainMask(segFile,paramsDict)

	## for this sub, gather all the data and labels for the SVM (also for specific ROI)
	#use mask based on union of all fold TT masks parN967_pre4_clndbold_perception_Mega_TTloop1p00GLM_RMSE_Nsigfolds.nii.gz
	#ttmsk = os.path.join(firstlevelDir,paramsDict['GLMMF_rslts_fmriDir'].replace('???',isub.replace('par','')),isub+'_'+paramsDict['preprocTag']+'_clndbold_perception_Mega_TT'+paramsDict['TT_mode']+[*paramsDict['TT_window'].keys()][0]+'GLM_RMSE_Nsigfolds'+'.nii.gz')
	#svmdata, pctTTvox, nvoxused = myfuncs.collectData4classification(currROI,firstlevelDir,isub,paramsDict,loops2include,mf_flag)
	subj_data = pd.DataFrame(data=None, index=None, columns=['subject', 'roi_group','nVox', 'offdiagmean_day1', 'ondiagmean_day1', 'offdiagmean_day2', 'ondiagmean_day2',
	'offdiagmin_day1', 'offdiagmax_day1', 'ondiagmin_day1', 'ondiagmax_day1', 'offdiagmin_day2', 'offdiagmax_day2', 'ondiagmin_day2', 'ondiagmax_day2',
	'offdiagstd_day1','ondiagstd_day1', 'offdiagstd_day2', 'ondiagstd_day2', 'day1_fullmean', 'day2_fullmean', 'day1_fullmin', 'day2_fullmin',
	'day1_fullmax', 'day2_fullmax', 'day1_fullstd', 'day2_fullstd'])

	plot_data = pd.DataFrame(data=None, index=None, columns=['subject', 'roi_group', 'day', 'corr_diff'])

	#initialize arrays to hold the diagonal differences for each subject and the voxel statistics for each
	d1_onminusoff = []
	d2_onminusoff = []
	voxelnums = []
	pvalues = np.empty(0)
	for isub in subject_ids:
		## init placeholder for this outdat (subXroiXscores)
		#OutData = pd.DataFrame(data = None, index = None ,columns=['subject','roi_group','pctTTvox','nVoxUsed','LRconfXTTactCorr','LRconfXTTtrsmCorr','LRconfXCRactCorr','LRconfXCRtrsmCorr','CVavgAcc'])
		## load the gray matter mask for this sub
		try:
			#maskFile = glob.glob(os.path.join(firstlevelDir,paramsDict['cleanfmriDir'].replace('???',isub.replace('par','')),isub+'_'+'T1W_dsegbrain'+'.nii.gz')) #parB355Schaefer40017.nii.gz
			submaskFile = glob.glob(os.path.join(firstlevelDir,paramsDict['cleanfmriDir'].replace('???',isub.replace('par','')),isub+'Schaefer40017dwn'+'.nii.gz')) #parB355Schaefer40017.nii.gz
			submask_data = nib.load(submaskFile[0])
		except:
			if paramsDict['debugMode']:
				print('couldnt find gray fmri mask for '+isub)


		currROI = myfuncs.singleMaskFromMultipleV2(root_path,roiKeyName,iroigroup,submask_data,paramsDict)

		## grab the loop ids for the 6 heard in all expo and rexpo runs
		eventFile = glob.glob(os.path.join(firstlevelDir,paramsDict['eventFilesPathLoop'].replace('???',isub.replace('par','')),paramsDict['EventNames']['expoFLL']['run-1'],'*.txt'))
		runEvent_dct = myfuncs.createEventTimingStimDct(paramsDict,eventFile,paramsDict['nLoopReps']) 
		loops2include = [*runEvent_dct.keys()]

		if paramsDict['useTTGLM_maps']:
			#use mask based on union of all fold TT masks parN967_pre4_clndbold_perception_Mega_TTloop1p00GLM_RMSE_Nsigfolds.nii.gz
			ttmsk = os.path.join(firstlevelDir,paramsDict['GLMMF_rslts_fmriDir'].replace('???',isub.replace('par','')),isub+'_'+paramsDict['preprocTag']+'_clndbold_perception_Mega_TT'+paramsDict['TT_mode']+[*paramsDict['TT_window'].keys()][0]+'GLM_RMSE_Nsigfolds'+'.nii.gz')
		elif paramsDict['useONGLM_maps']:
			ttmsk = os.path.join(firstlevelDir,paramsDict['GLMMF_rslts_fmriDir'].replace('???',isub.replace('par','')),isub+'_'+paramsDict['preprocTag']+'_clndbold_perception_Mega_ON'+paramsDict['ON_mode']+'GLM_TVAL_Nsigfolds'+'.nii.gz')
		tmpvoxinfo = pd.read_csv(os.path.join(firstlevelDir,paramsDict['RSAROI_fmriDir'].replace('???',isub.replace('par','')),paramsDict['preprocTag']+useTTflag+'_'+roiKeyName[2]+'_nVox.csv'))		
		
		#svmdata, pctTTvox, nvoxused = myfuncs.collectData4classification(currROI,firstlevelDir,isub,paramsDict,loops2include,mf_flag, ttmsk)
		
		#add number of voxels for this subject in this roi to roi level collection of voxel numbers
		
		try:
			nvoxused = [*tmpvoxinfo.loc[np.where(tmpvoxinfo['roi_group']==iroigroup)]['nVoxUsed']][0]
			pctTTvox = [*tmpvoxinfo.loc[np.where(tmpvoxinfo['roi_group']==iroigroup)]['pctTTvox']][0]
		except:
			#print('No voxels found in'+iroigroup+'for sub: '+isub)
			voxelnums.append(np.nan)
			pctTTvox = 0

		voxelnums.append(nvoxused)
		print('found '+str(nvoxused)+' voxels for '+isub)
		if  nvoxused>=paramsDict['minNVox4RSA']: #
			d1path = os.path.join(firstlevelDir,paramsDict['RSAROI_indroi_fmriDir'].replace('???',isub.replace('par','')),paramsDict['preprocTag']+useTTflag+'_'+roiKeyName[2]+iroigroup+'_D1_corrmat.csv')
			d2path = os.path.join(firstlevelDir,paramsDict['RSAROI_indroi_fmriDir'].replace('???',isub.replace('par','')),paramsDict['preprocTag']+useTTflag+'_'+roiKeyName[2]+iroigroup+'_D2_corrmat.csv')
			

			#run a try, except to figure out the problem
			d1_df = pd.read_csv(d1path)
			d2_df = pd.read_csv(d2path)

			#d1_df_all = []
			#get avg heatmap for day 1
			#d1_df_all.append(d1_df)
			

			#D1 column 
			d1_ttarray = d1_df.drop(columns=['Unnamed: 0']).to_numpy() #this actually makes it so columns are arrays (1 late stims corr with each early stim)

			#on diagonal average
			d1_diag = np.diag(d1_ttarray)
			ondiag_avgd1 = np.mean(d1_diag)

			#off diagonal average
			#get upper and lower diags, ignore the main diag
			d1_upper = np.triu(d1_ttarray, 1)
			d1_lower = np.tril(d1_ttarray, -1)
			#flatten arrays
			d1_upperflat = np.ndarray.flatten(d1_upper)
			d1_lowerflat = np.ndarray.flatten(d1_lower)
			#take out nonzeros
			d1_upperflat = d1_upperflat[np.nonzero(d1_upperflat)]
			d1_lowerflat = d1_lowerflat[np.nonzero(d1_lowerflat)]
			#combine upper and lower triangles to take mean
			offdiag_avgd1 = np.mean(np.concatenate((d1_upperflat, d1_lowerflat)))


			#D2 column
			d2_ttarray = d2_df.drop(columns=['Unnamed: 0']).to_numpy()

			#on diagonal average
			d2_diag = np.diag(d2_ttarray)
			ondiag_avgd2 = np.mean(d2_diag)

			#off diagonal average
			#get upper and lower diags, ignore the main diag
			d2_upper = np.triu(d2_ttarray, 1)
			d2_lower = np.tril(d2_ttarray, -1)
			#flatten arrays
			d2_upperflat = np.ndarray.flatten(d2_upper)
			d2_lowerflat = np.ndarray.flatten(d2_lower)
			#take out nonzeros
			d2_upperflat = d2_upperflat[np.nonzero(d2_upperflat)]
			d2_lowerflat = d2_lowerflat[np.nonzero(d2_lowerflat)]
			#combine upper and lower triangles to take mean
			offdiag_avgd2 = np.mean(np.concatenate((d2_upperflat, d2_lowerflat)))

			d1_onminusoff.append(ondiag_avgd1 - offdiag_avgd1)
			d2_onminusoff.append(ondiag_avgd2 - offdiag_avgd2)

			#
			
			subj_data = subj_data.append({'subject': isub, 'roi_group': iroigroup,'nVox':nvoxused, 'offdiagmean_day1': offdiag_avgd1, 'ondiagmean_day1': ondiag_avgd1, 'offdiagmean_day2': offdiag_avgd2, 'ondiagmean_day2': ondiag_avgd2,
			'offdiagmin_day1': min(np.concatenate((d1_upperflat, d1_lowerflat))), 'offdiagmax_day1': max(np.concatenate((d1_upperflat, d1_lowerflat))), 'ondiagmin_day1': min(d1_diag), 'ondiagmax_day1': max(d1_diag),
			'offdiagmin_day2': min(np.concatenate((d2_upperflat, d2_lowerflat))), 'offdiagmax_day2': max(np.concatenate((d2_upperflat, d2_lowerflat))), 'ondiagmin_day2': min(d2_diag), 'ondiagmax_day2': max(d2_diag),
			'offdiagstd_day1': np.std(np.concatenate((d1_upperflat, d1_lowerflat))),'ondiagstd_day1': np.std(d1_diag), 'offdiagstd_day2': np.std(np.concatenate((d2_upperflat, d2_lowerflat))), 'ondiagstd_day2': np.std(d2_diag),
			'day1_fullmean': np.mean(d1_ttarray.flatten()), 'day2_fullmean': np.mean(d2_ttarray.flatten()), 'day1_fullmin': min(d1_ttarray.flatten()), 'day2_fullmin': min(d2_ttarray.flatten()),
			'day1_fullmax': max(d1_ttarray.flatten()), 'day2_fullmax': max(d2_ttarray.flatten()), 'day1_fullstd': np.std(d1_ttarray.flatten()), 'day2_fullstd': np.std(d2_ttarray.flatten())}, ignore_index=True)

			plot_data = plot_data.append({'subject': isub, 'roi_group': iroigroup, 'day': 'Day 1', 'corr_diff': (ondiag_avgd1 - offdiag_avgd1)}, ignore_index=True)
			plot_data = plot_data.append({'subject': isub, 'roi_group': iroigroup, 'day': 'Day 2', 'corr_diff': (ondiag_avgd2 - offdiag_avgd2)}, ignore_index=True)
		else:
			subj_data = subj_data.append({'subject': isub, 'roi_group': iroigroup,'nVox':np.nan, 'offdiagmean_day1': np.nan, 'ondiagmean_day1': np.nan, 'offdiagmean_day2': np.nan, 'ondiagmean_day2': np.nan,
			'offdiagmin_day1': np.nan, 'offdiagmax_day1': np.nan, 'ondiagmin_day1': np.nan, 'ondiagmax_day1': np.nan,
			'offdiagmin_day2': np.nan, 'offdiagmax_day2': np.nan, 'ondiagmin_day2': np.nan, 'ondiagmax_day2': np.nan,
			'offdiagstd_day1': np.nan,'ondiagstd_day1': np.nan, 'offdiagstd_day2': np.nan, 'ondiagstd_day2':np.nan,
			'day1_fullmean': np.nan, 'day2_fullmean': np.nan, 'day1_fullmin': np.nan, 'day2_fullmin': np.nan,
			'day1_fullmax': np.nan, 'day2_fullmax': np.nan, 'day1_fullstd': np.nan, 'day2_fullstd': np.nan}, ignore_index=True)

			#plot_data = plot_data.append({'subject': isub, 'roi_group': iroigroup, 'day': 'Day 1', 'corr_diff': np.nan}, ignore_index=True)
			#plot_data = plot_data.append({'subject': isub, 'roi_group': iroigroup, 'day': 'Day 2', 'corr_diff': np.nan}, ignore_index=True)

	if len(np.unique(plot_data['subject']))==len(subject_ids):
		#only plot/test and ROI if it has data from all subjects.

		tmp_ttestdat = plot_data.loc[np.where(plot_data['roi_group']==iroigroup)]
		tmp_ttestdat = tmp_ttestdat.pivot(index='subject', columns='day', values='corr_diff').reset_index()
		#need to make sure things line up even if missing sub for an ROI
		ttestresult, pvalue = stats.ttest_rel([*tmp_ttestdat['Day 2']],[*tmp_ttestdat['Day 1']]) 
		
		if paramsDict[roiKeyName][iroigroup][0]>2000:
			trnscoords = (paramsDict[roiKeyName][iroigroup][0]-2000) + 200

		elif paramsDict[roiKeyName][iroigroup][0]>1000:
			trnscoords = (paramsDict[roiKeyName][iroigroup][0]-1000)

		else:
			#sub-cortical skip
			trnscoords = paramsDict[roiKeyName][iroigroup][0]
		
		indices = np.where(mask_data.get_fdata().astype('int') == trnscoords)
		outtval_vol[indices] = int(ttestresult)
		if pvalue < 0.05:
			if ttestresult>0:
				outsigmask_vol[indices] = 1
			elif ttestresult<0:
				outsigmask_vol[indices] = -1

		else:
			outsigmask_vol[indices] = 0
		

		suboutfile1 = os.path.join(firstlevelDir,paramsDict['RSAgroupDir_indroi'],paramsDict['preprocTag']+useTTflag+'_'+roiKeyName[2]+iroigroup+'_indroi_corrdata.csv')
		suboutfile3 = os.path.join(firstlevelDir,paramsDict['RSAgroupDir_indroi'],paramsDict['preprocTag']+useTTflag+'_'+roiKeyName[2]+iroigroup+'_indroi_plotdata.csv')
		suboutfile4 = os.path.join(firstlevelDir,paramsDict['RSAgroupDir_indroi'],paramsDict['preprocTag']+useTTflag+'_'+roiKeyName[2]+iroigroup+'_indroiplot.pdf')

		subj_data.to_csv(suboutfile1)
		plot_data.to_csv(suboutfile3)
		#pdb.set_trace()
		#could be good to save out mean value for day 2 and for day 1, and the min and max for each, std
		#create csv file, each row an ROI, column for T, column for p, number of values day 1 and day 2 (sanity check)
		#ask Ben which one to use, one takes track off and on diagonal values, one just looks at entire matrix
		#ask Ben if this makes sense; some of these statistics are for individual subs and some are group, think I'm just taking values from last subjects data?

		TTest_Data = TTest_Data.append({'roi_group': iroigroup,'nsubs': len(np.unique(tmp_ttestdat['subject'])) ,'maxvox': max(voxelnums), 'minvox': min(voxelnums), 'avgvoxused': np.mean(voxelnums), 'stdvox': np.std(voxelnums), 'd1_diffmax': max(d1_onminusoff), 'd1_diffmin': min(d1_onminusoff), 'd1_diffavg': np.mean(d1_onminusoff),
		'd1_diffstd': np.std(d1_onminusoff), 'd2_diffmax': max(d2_onminusoff), 'd2_diffmin': min(d2_onminusoff), 'd2_diffavg': np.mean(d2_onminusoff), 'd2_diffstd': np.std(d2_onminusoff), 'tscore': ttestresult, 'pvalue': pvalue}, ignore_index=True)
		
		fig, ax = plt.subplots(figsize=(12,12))
		sns.lineplot(data=plot_data, x=plot_data['day'], y=plot_data['corr_diff'], hue=plot_data['subject'], marker='o')
		sns.boxplot(data=plot_data, x=plot_data['day'], y=plot_data['corr_diff'])
		ax.get_legend().remove()
		plt.title(iroigroup + ' Subject Correlation Differences', fontsize=15)
		#plotting.plot_stat_map(roiscatterplot, display_mode='x', threshold=volfiles[ivol]['threshold'],vmax=volfiles[ivol]['vmax'],cut_coords=[1,34,55], title='',colorbar=volfiles[ivol]['colorbar'])
		#plt.savefig(os.path.join(rootDir,isub+'_'+os.path.basename(ivol).replace('.nii.gz','_R.png')))
		plt.savefig(suboutfile4, format='pdf')

	#pdb.set_trace()
	#%pd.concat([d1_df_all[0], d1_df_all[1]]).groupby(level=0).mean()

'''
'offdiagmean_day1': offdiag_avgd1, 'ondiagmean_day1': ondiag_avgd1, 'offdiagmean_day2': offdiag_avgd2, 'ondiagmean_day2': ondiag_avgd2,
	'offdiagmin_day1': min(np.concatenate((d1_upperflat, d1_lowerflat))), 'offdiagmax_day1': max(np.concatenate((d1_upperflat, d1_lowerflat))), 'ondiagmin_day1': min(d1_diag), 'ondiagmax_day1': max(d1_diag),
	'offdiagmin_day2': min(np.concatenate((d2_upperflat, d2_lowerflat))), 'offdiagmax_day2': max(np.concatenate((d2_upperflat, d2_lowerflat))), 'ondiagmin_day2': min(d2_diag), 'ondiagmax_day2': max(d2_diag),
	'offdiagstd_day1': np.std(np.concatenate((d1_upperflat, d1_lowerflat))),'ondiagstd_day1': np.std(d1_diag), 'offdiagstd_day2': np.std(np.concatenate((d2_upperflat, d2_lowerflat))), 'ondiagstd_day2': np.std(d2_diag),
	'day1_fullmean': np.mean(d1_ttarray.flatten()), 'day2_fullmean': np.mean(d2_ttarray.flatten()), 'day1_fullmin': min(d1_ttarray.flatten()), 'day2_fullmin': min(d2_ttarray.flatten()),
	'day1_fullmax': max(d1_ttarray.flatten()), 'day2_fullmax': max(d2_ttarray.flatten()), 'day1_fullstd': np.std(d1_ttarray.flatten()), 'day2_fullstd': np.std(d2_ttarray.flatten()),
'''
#min, max, average voxels used in df
#each row is an ROI so only extract group level things like those above
#save out to 'CLgroupDir' instead of 'CLROI_indroi'
suboutfile2 = os.path.join(firstlevelDir,paramsDict['RSAgroupDir'],paramsDict['preprocTag']+useTTflag+'_'+roiKeyName[2]+'_TTest_Results.csv')
#suboutfile1 = open(os.path.join(firstlevelDir,paramsDict['CLgroupDir'],paramsDict['preprocTag']+useTTflag+'_'+roiKeyName[2]+'indroi_corrdata.csv'),"w+")

pvaluearray = TTest_Data['pvalue'].to_numpy()

#from statsmodels.stats.multitest import fdrcorrection
#fdrcorrection(pvaluearray)
qvals = FDR_p(pvaluearray)

TTest_Data['qvalue'] = qvals.tolist()

out_tvals = nib.Nifti1Image(outtval_vol, mask_data.affine)  # create the volume image
hdr1 = out_tvals.header  # get a handle of the .nii file's header
hdr1.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
nib.save(out_tvals,os.path.join(firstlevelDir,paramsDict['RSAgroupDir'],paramsDict['preprocTag']+useTTflag+'_'+roiKeyName[2]+'TTest_Tvals.nii.gz'))

out_sigvals = nib.Nifti1Image(outsigmask_vol, mask_data.affine)  # create the volume image
hdr1 = out_sigvals.header  # get a handle of the .nii file's header
hdr1.set_zooms((dimsize[0], dimsize[1], dimsize[2]))
nib.save(out_sigvals,os.path.join(firstlevelDir,paramsDict['RSAgroupDir'],paramsDict['preprocTag']+useTTflag+'_'+roiKeyName[2]+'TTest_Sigvals.nii.gz'))

TTest_Data.to_csv(suboutfile2)
#output goal: plots for each ROI that show subject change and brain mask that shows paired T-test significance, make folder within CLresults for pairedttest plots
