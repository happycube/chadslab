from __future__ import print_function
from __future__ import division
import numpy as np
import xgboost as xgb
from datetime import  datetime
import difflib

data_test = {} # sl2.data_test
data_train = {} # sl2.data_train
data_comb = {} # sl2.data_comb

# command used to run this for .80130 LB:
# x_nt = sl2a.buildxgb(sl2.data_comb['binmatrix'][0:tlen], sl2.data_comb['data'][1933][0:tlen], num_rounds=9001)

# from ipython:
# reload(sl2a) ; sl2a.data_test = sl2.data_test; sl2a.data_train = sl2.data_train; sl2a.data_comb = sl2.data_comb

dateset = {73, 75, 156, 157, 158, 159, 166, 168, 169, 176, 178, 179, 204, 217}
datecache = {}
	
numbins = 100 - 1
numsbins = 2000 - 1

def do_binify(data, i, exclude = None):
	length = len(data['data'][0])
	tlen = len(data_comb['data'][1933])
	
	thisdata = data['data'][i]

	if (i in dateset):
		print(i, "date")
		intdata = np.zeros(length, dtype=np.float32)
			
		for item in range(0, length):
			if (len(thisdata[item]) > 1): 
#				print(i, item, thisdata[item].split('\"')[0])
				date = thisdata[item].split('\"')[0]

				if date not in datecache:
					datecache[date] = datetime.strptime(date, '%d%b%y:%H:%M:%S').isocalendar()[1]
					datecache[date] += (datetime.strptime(date, '%d%b%y:%H:%M:%S').isocalendar()[0] * 52)

				intdata[item] = datecache[date] 
#				intdata[item] = datetime.strptime(date, '%d%b%y:%H:%M:%S').isocalendar()[1]

		thisdata = intdata 

	elif (thisdata.dtype.type == np.string_):
		print(i, "string")
		intdata = np.zeros(length, dtype=np.float32)
		
		# Must restrict this phase to 0:tlen to keep w/time machine rule!	
		for item in range(exclude, tlen):
			thisdata[item] = thisdata[item].split('\"')[0]

		sorteddata = np.unique(thisdata[exclude:tlen])

		sortedhasheddata = {}
		for k in range(exclude, len(sorteddata)):
			sortedhasheddata[sorteddata[k]] = k

		print(len(sorteddata))

		# now bin everything
		for item in range(0, length):
			if item >= tlen:
				thisdata[item] = thisdata[item].split('\"')[0]
			
			k = thisdata[item]

			if k not in sortedhasheddata:
				for ki in range(0, len(sorteddata)):
					if k >= sorteddata[ki]: 
						k = sorteddata[ki]
						break

			if k not in sortedhasheddata:
				intdata[item] = np.nan
			else:	
				intdata[item] = sortedhasheddata[k]
	
		thisdata = intdata 
	else:
		print(i, "reg")

#	print(i, len(thisdata))
	bins = np.unique(np.sort(thisdata[exclude:tlen])[::tlen//numbins])

	if (i != 0) and (thisdata.dtype.type != np.string_) and (len(thisdata) == length) and (len(bins) > 1):
		data['bins'][i] = bins 
		data['binned'][i] = np.digitize(thisdata, bins) 
		#	print(i, bins, data['binned'][i])

def binify(data, exclude = 0):
	length = len(data['data'][0])
	
	tlen = len(data_comb['data'][1933])

	data['bins'] = {}
	data['binned'] = np.empty((len(data['data'].keys()), length), dtype=np.float32)
#	data['bin_values'] = np.empty((len(data['data'].keys()), length), dtype=np.int16)

	keyset = data['data'].keys()

        keyset.remove(1933)
#        keyset.remove(200)
        keyset.remove(0)

	datecache = {}

	for i in keyset:	
		do_binify(data, i, exclude = exclude)

def feature_words(data, evallen = None):
	tlen = len(data['data'][1933])

	jobs = data['data'][491]
	numjobs = len(jobs)

	begin = evallen if (evallen != None) else 0

	# create 4?letter word bins
	allwords = []
	allwords.append("") # set aside 0 for no word 

	for i in range(begin, tlen):
		for w in jobs[i].split(" "):
			if len(w) >= 4:
				allwords.append(w.split('\"')[0])	

	#allwords = np.unique(allwords)
	allwords = np.unique(np.sort(allwords[::len(allwords)//numbins]))
#	bins = np.unique(np.sort(thisdata[exclude:tlen])[::tlen//numbins])

	wordhash = {}
	for i in range(0, len(allwords)):
		wordhash[allwords[i]] = i

	# figure out longest title
	maxcount = -1
	for i in range(0, numjobs):
		count = len(jobs[i].split(" "))
		maxcount = count if (count > maxcount) else maxcount

#	if maxcount > 3:
#		maxcount = 3

	print(maxcount)

	# now build the actual job matrix
	jobmatrix = np.zeros((numjobs, maxcount), dtype=np.float32)
	thisone = []
					
	worddiff = np.zeros(len(allwords), dtype=np.float32) 

	for i in range(0, numjobs):
		splitjob = jobs[i].split(" ")
		thisone = []
	
		for w in range(0, maxcount):
			jobmatrix[i][w] = np.nan 

		for w in range(0, len(splitjob)):
			if w < maxcount and len(splitjob[w]) >= 4:
				cutword = splitjob[w].split('\"')[0]
				if cutword in wordhash: 
#					jobmatrix[i][w] = wordhash[cutword]
					thisone.append(wordhash[cutword])
				else:
			#		for j in range(1, len(allwords)):
#						print(cutword, allwords[j], difflib.SequenceMatcher(a=allwords[j], b=cutword).ratio())
			#			worddiff[j] = difflib.SequenceMatcher(a=allwords[j], b=cutword).ratio()

			#		thisone.append(np.argmax(worddiff))

					for j in range(1, len(allwords)):
						if allwords[j] >= cutword:
#							print(j, allwords[j], cutword)
#							jobmatrix[i][w] = j
							thisone.append(j)
							break

			count = 0
			#for v in np.sort(thisone):
			for v in thisone:
				jobmatrix[i][count] = v
				count = count + 1

#		if jobmatrix[i][0] != 0:
#			print(i, jobmatrix[i])

	jobmatrix = jobmatrix.transpose()

	print(data['binned'][491][begin:tlen])
	print(jobmatrix[begin:tlen])

	return jobmatrix

	# orig
#	omatrix = np.zeros((numjobs, 1), dtype=np.int16)
#	for i in range(0, numjobs):
#		omatrix[i][0] = data['binned'][491][i] 

#	xgb_params = {"objective": "binary:logistic", "eta":.008, "subsample":0.7, "colsample_bytree":1.0, "max_depth": 4, "silent": 0, "min_child_weight":2}
#	dtrain = xgb.DMatrix(omatrix[begin:tlen], label=data['data'][1933][begin:tlen].astype(np.int16), missing=0)
#	evallist = [(dtrain,'train')]

#	if begin != 0:
#		dtest = xgb.DMatrix(omatrix[0:begin], label=data['data'][1933][0:begin].astype(np.int16), missing=0)
#		evallist = [(dtest,'eval'), (dtrain,'train')]

#	num_rounds = 100
	
#	plst = xgb_params.items()
#	plst += [('eval_metric', 'auc')]
#	gbdt = xgb.train(plst, dtrain, num_rounds, evallist)

	# new	
	xgb_params = {"objective": "binary:logistic", "eta":.010, "subsample":0.70, "colsample_bytree":0.70, "max_depth": 4, "silent": 0, "min_child_weight":2}
	dtrain = xgb.DMatrix(jobmatrix[begin:tlen], label=data['data'][1933][begin:tlen].astype(np.int16), missing=0)
	evallist = [(dtrain,'train')]

	if begin != 0:
		dtest = xgb.DMatrix(jobmatrix[0:begin], label=data['data'][1933][0:begin].astype(np.int16), missing=0)
		evallist = [(dtest,'eval'), (dtrain,'train')]

	num_rounds = 150
	
	plst = xgb_params.items()
	plst += [('eval_metric', 'auc')]
	gbdt = xgb.train(plst, dtrain, num_rounds, evallist)
	
	predict = gbdt.predict(xgb.DMatrix(jobmatrix, missing=0))
#	print(predict)

#	data['binned']['492x'] = np.empty(numjobs, np.float32)
	#for i in range(0, numjobs):
	#	data['binned'][492][i] = (predict[i] * 10000)

	return predict

def bin2mat(data, keyset = None, exclude = None, extras = None):

	if keyset == None:
		keyset = np.sort(data['bins'].keys())

#	for e in exclude:
       # 	keyset.remove(e)
#	print(len(keyset))	

	if exclude != None:
		keyset2 = []
		for i in range(0, len(keyset)):
			if keyset[i] not in exclude:
				keyset2.append(keyset[i])

		keyset = keyset2

	numkeys = len(keyset)

	if extras:
		for nf in extras:
			numkeys += nf.shape[0] 

	length = len(data['data'][0])
	data['binmatrix'] = np.empty((numkeys, length), dtype=np.int16)

	count = 0
	for i in keyset:
		print(i, count)
		data['binmatrix'][count] = data['binned'][i]	
		count = count + 1

#	print(data['binmatrix'].shape)	
	if extras:
		for nf in extras:
			for i in range(0, nf.shape[0]):
				print(i, count)
				data['binmatrix'][count] = nf[i]	
				count = count + 1

#		for i in range(0, nf.shape[0]):	
#			numkeys += nf.shape[1] 
	
	data['binmatrix'] = data['binmatrix'].transpose()
#	for nf in extras:

def buildxgb(prob_matrix, tgt, test_matrix = None, test_tgt = None, num_rounds = 10):
	ts = datetime.now()

#	xgb_params = {"objective": "binary:logistic", "eta":.001, "subsample":0.6, "colsample_bytree":0.6, "max_depth": 14, "silent": 0}
#	xgb_params = {"objective": "binary:logistic", "eta":.002, "subsample":0.7, "colsample_bytree":0.7, "max_depth": 10, "silent": 0}
	xgb_params = {"objective": "binary:logistic", "eta":.01, "subsample":0.8, "colsample_bytree":0.7, "max_depth": 12, "silent": 0}
	xgb_params = {"objective": "binary:logistic", "eta":.009, "subsample":0.8, "colsample_bytree":0.7, "max_depth": 11, "silent": 0}
	xgb_params = {"objective": "binary:logistic", "eta":.009, "subsample":0.5, "colsample_bytree":0.7, "max_depth": 11, "silent": 0}
	xgb_params = {"objective": "binary:logistic", "eta":.009, "subsample":0.5, "colsample_bytree":0.7, "max_depth": 11, "silent": 0}
	xgb_params = {"objective": "binary:logistic", "eta":.01, "subsample":0.5, "colsample_bytree":0.7, "max_depth": 11, "silent": 0}
	xgb_params = {"objective": "binary:logistic", "eta":.01, "subsample":0.5, "colsample_bytree":1.0, "max_depth": 11, "silent": 0}
	xgb_params = {"objective": "binary:logistic", "eta":.009, "subsample":0.5, "colsample_bytree":1.0, "max_depth": 11, "silent": 0}
	
	xgb_params = {"objective": "binary:logistic", "eta":.009, "subsample":0.5, "colsample_bytree":1.0, "max_depth": 11, "silent": 0, "min_child_weight":2}

	# actual .8013 one
	xgb_params = {"objective": "binary:logistic", "eta":.009, "subsample":0.8, "colsample_bytree":0.7, "max_depth": 11, "silent": 0}

	# +++	
	xgb_params = {"objective": "binary:logistic", "eta":.008, "subsample":0.65, "colsample_bytree":0.7, "max_depth": 11, "silent": 0, "min_child_weight":2}

	xgb_params = {"objective": "binary:logistic", "eta":.008, "subsample":0.60, "colsample_bytree":0.70, "max_depth": 11, "silent": 0, "min_child_weight":2}
	xgb_params = {"objective": "binary:logistic", "eta":.008, "subsample":0.60, "colsample_bytree":0.70, "max_depth": 11, "silent": 0, "min_child_weight":2}
	xgb_params = {"objective": "binary:logistic", "eta":.009, "subsample":0.8, "colsample_bytree":0.7, "max_depth": 11, "silent": 0, "nthreads":4}
	xgb_params = {"objective": "binary:logistic", "eta":.009, "subsample":0.8, "colsample_bytree":0.7, "max_depth": 14, "silent": 0, "nthreads":8, "seed":12345678}

	dtrain = xgb.DMatrix(prob_matrix, label=tgt.astype(np.float32), missing=np.nan)

	dtest = None
	evallist = [(dtrain,'train')]

	if test_matrix != None:
		dtest = xgb.DMatrix(test_matrix, label=test_tgt.astype(np.float32), missing=np.nan)
		evallist = [(dtest,'eval'), (dtrain,'train')]
	
	plst = xgb_params.items()
	plst += [('eval_metric', 'auc')]
	gbdt = xgb.train(plst, dtrain, num_rounds, evallist)

	te = datetime.now()
	print('elapsed time: {0}'.format(te-ts))

	return gbdt

def dobuild():
	x = sl2a.buildxgb(sl2.data_comb['binmatrix'][15000:tlen], sl2.data_comb['data'][1933][15000:tlen], sl2.data_comb['binmatrix'][0:15000], sl2.data_comb['data'][1933][0:15000]) 

	return x

def dosub(x):
	preds = x.predict(xgb.DMatrix(data_comb['binmatrix'], missing=np.nan))

	tlen = len(data_comb['data'][1933])

	of = open('sub', 'w')
	of.write('ID,target\n')
	for i in range(tlen,len(preds)):
		of.write('%d,%f\n' % (data_comb['data'][0][i], preds[i]))
	of.close()

def combine():
        global data_train, data_test, data_comb 

#	data_comb = {} 

	data_comb['data'] = {}

        for i in data_test['data'].keys():
#		print i
                data_comb['data'][i] = np.concatenate((data_train['data'][i], data_test['data'][i]))

        # target value
        data_comb['data'][1933] = data_train['data'][1933]

dtrain = None

# unfinished
def run_seq(data):
	global dtrain

	ts = datetime.now()
	
#x = sl2a.buildxgb(sl2.data_comb['binmatrix'][tlen-7500:tlen], sl2.data_comb['data'][1933][tlen-7500:tlen], num_rounds=800)
	xgb_params = {"objective": "binary:logistic", "eta":.015, "subsample":0.60, "colsample_bytree":0.70, "max_depth": 11, "silent": 0, "min_child_weight":2}

	dtrain = xgb.DMatrix(data['binmatrix'], label=data['data'][1933].astype(np.float32), missing=np.nan)
	
	dtest = None
	evallist = [(dtrain,'train')]

	plst = xgb_params.items()
	plst += [('eval_metric', 'auc')]
	gbdt = xgb.train(plst, dtrain, 700, evallist)

	te = datetime.now()
	print('elapsed time: {0}'.format(te-ts))

