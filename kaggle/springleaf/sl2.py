#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import sys
import mmap

import numpy as np
import scipy as sp

import cPickle as pickle

def ParseData(fname, ctx = {}):
	f = open(fname, "r+b")
	mm = mmap.mmap(f.fileno(), 0)

	print(len(mm))

	ctx['nrecords'] = {} 
	ctx['data'] = {} 
	ctx['floatdata'] = {}
	ctx['name'] = {}

	lens = np.fromstring(mm[0:16], 'uint64', 2)
	print(lens)

	nfields = lens[0]
	nrecords = lens[1]
	
	ilens = np.empty(nfields) 

	for i in range(0, nfields):
		tgt = (i * 40) + 32

		tgtinfo = np.fromstring(mm[tgt+16:tgt+40], 'uint64', 3)
		tgtbegin = tgtinfo[0]
		tgtend = tgtinfo[0] + (tgtinfo[1] * nrecords)

		ilens[i] = tgtlen = tgtinfo[1]
 
		print(i, mm[tgt:tgt+16], np.fromstring(mm[tgt+16:tgt+40], 'uint64', 3))

		ctx['name'][i] = mm[tgt:tgt+16].split('\x00')[0]

		predata = None

		if tgtinfo[2] == 0:  # integer
			if tgtinfo[1] == 2: 
				predata = np.fromstring(mm[tgtbegin:tgtend], 'int16', nrecords).astype(np.float)
	#			floatdata[i] = gpuarray.to_gpu(predata)
				ctx['data'][i] = (np.fromstring(mm[tgtbegin:tgtend], 'int16', nrecords)) 
			elif tgtinfo[1] == 4: 
				predata = np.fromstring(mm[tgtbegin:tgtend], 'int32', nrecords).astype(np.float)
	#			floatdata[i] = gpuarray.to_gpu(predata)
				ctx['data'][i] = (np.fromstring(mm[tgtbegin:tgtend], 'int32', nrecords)) 
			elif tgtinfo[1] == 8: 
				predata = np.fromstring(mm[tgtbegin:tgtend], 'int64', nrecords).astype(np.float)
	#			floatdata[i] = gpuarray.to_gpu(predata)
				ctx['data'][i] = (np.fromstring(mm[tgtbegin:tgtend], 'int64', nrecords)) 

		if tgtinfo[2] == 1: # float
			predata = np.fromstring(mm[tgtbegin:tgtend], 'float32', nrecords)
#			floatdata[i] = gpuarray.to_gpu(predata)
			ctx['data'][i] = predata # gpuarray.to_gpu(predata)

		if tgtinfo[2] == 2: # string
			ctx['data'][i] = np.fromstring(mm[tgtbegin:tgtend], (np.str_, tgtlen), nrecords)

	return ctx
#	return (ctx['nrecords'], ctx['name'], ctx['data'], ctx['floatdata'], ctx['stringdata'])

data_train = {} 
data_test = {} 

data_comb = {}

def loaddata(datafile, picklefile):
	data = {}
	try:
		pfd = open(picklefile, 'rb')
		data = pickle.load(pfd)
		pfd.close()
	except:
		data = ParseData(datafile)
		pfd = open(picklefile, 'wb+')
		pickle.dump(data, pfd, protocol=2) 
		pfd.close()

	return data
#	ftest = open('test.p', 'rb')

def combine():
	global data_train, data_test, data_comb

	for i in data_test['data'].keys():
		data_comb['data'][i] = np.concatenate(sl2.data_train['data'][i], sl2.data_test['data'][i])

	# target value
	data_comb['data'][1933] = data_train['data'][1933]

def startup():
	global data_train, data_test
	
	data_train = loaddata("../input/output.bin", "train.p")
	print(data_train['data'][1933])
	data_test = loaddata("../input/output-test.bin", "test.p")

	# XXX:  why do i need to reload after doing a fresh load!?	
	data_train = loaddata("../input/output.bin", "train.p")
	print(data_train['data'][1933])
#	print(data_test['data'][1933])
	
	

#	ftest = open('test.p', 'rb')
	

#	train_context = {}
#	train = loader.ParseData("../input/output.bin", train_context)
#	train_prob = sort.runprob(train)
#	train_prob_matrix = sort.build_probmatrix(train_prob, len(train[2][0]) - 15000)
#	train_prob_matrix = sort.applyprob(train, train_prob, all=True)
	
#	test_context = {}
#	test = loader.ParseData("../input/output-test.bin", test_context)
#	test_prob_matrix = sort.applyprob(test, train_prob)
	
#	test_context = {}
#	test = loader.ParseData("output-test.bin", test_context)
#	test_prob = sort.runprob(test)
#	test_prob_matrix = sort.build_probmatrix(test_prob, len(test[2][0]))






