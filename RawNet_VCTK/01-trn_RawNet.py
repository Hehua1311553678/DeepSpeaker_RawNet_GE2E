import os
import numpy as np
np.random.seed(1016)
import yaml
import queue
import struct
import pickle as pk
from multiprocessing import Process
from threading import Thread
from tqdm import tqdm
import time
from time import sleep

from keras.utils import multi_gpu_model, plot_model, to_categorical
from keras.optimizers import *
from keras.models import Model
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from model_RawNet_pre_train import get_model as get_model_pretrn
from model_RawNet import get_model

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def cos_sim(a,b):
	return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def simple_loss(y_true, y_pred):
	return K.mean(y_pred)

def zero_loss(y_true, y_pred):
	return 0.5 * K.sum(y_pred, axis=0)

def compose_spkFeat_dic(lines, model, f_desc_dic, base_dir):
	'''
	Extracts speaker embeddings from a given model
	=====

	lines: (list) A list of strings that indicate each utterance
	model: (keras model) DNN that extracts speaker embeddings,
		output layer should be rmoved(model_pred)
	f_desc_dic: (dictionary) A dictionary of file objects
	'''
	dic_spkFeat = {}
	for line in tqdm(lines, desc='extracting spk feats'):
		k, f, p = line.strip().split(' ')
		p = int(p)
		if f not in f_desc_dic:
			f_tmp = f #================================================2020/04/19 23:20
			# f_tmp = '/'.join([base_dir, f])
			# f_tmp = f_tmp.replace('/', '\\')
			f_desc_dic[f] = open(f_tmp, 'rb')

		f_desc_dic[f].seek(p)
		l = struct.unpack('i', f_desc_dic[f].read(4))[0]# number of samples of each utterance
		utt = np.asarray(struct.unpack('%df'%l, f_desc_dic[f].read(l * 4)), dtype=np.float32)# read binary utterance 
		spkFeat = model.predict(utt.reshape(1,-1,1))[0]# extract speaker embedding from utt
		dic_spkFeat[k] = spkFeat

	return dic_spkFeat

def make_spkdic(lines):
	'''
	Returns a dictionary where
		key: (str) speaker name
		value: (int) unique integer for each speaker
	'''
	idx = 0
	dic_spk = {}
	list_spk = []
	for line in lines:
		k, f, p = line.strip().split(' ')
		spk = k.split('/')[1]
		if spk not in dic_spk:
			dic_spk[spk] = idx
			list_spk.append(spk)
			idx += 1
	return (dic_spk, list_spk)

def compose_batch(lines, f_desc_dic, dic_spk, nb_samp, base_dir):
	'''
	Compose one mini-batch using utterances in `lines'

	nb_samp: (int) duration of utterance at train phase.
		Fixed for each mini-batch for mini-batch training.
	'''
	batch = []
	ans = []
	num_lines = len(lines) #=================================================================2020/04/13 21:44
	for line in lines:
		k, f, p = line.strip().split(' ')
		# line举例：'train/S0002/BAC009S0002W0122.wav aishell1/scp/train/train_wav_0_pe.ark 0\n'
		# ans.append(dic_spk[k.split('/')[1]])#=====================================2020/04/13 23:43
		p = int(p)
		if f not in f_desc_dic:
			f_tmp = f# ================================================2020/04/19 23:20
			# f_tmp = '/'.join([base_dir, f])
			# f_tmp = f_tmp.replace('/', '\\')
			f_desc_dic[f] = open(f_tmp, 'rb')
			# f_desc_dic['aishell1/scp/train/train_wav_0_pe.ark'] = 对应打开的文件。

		f_desc_dic[f].seek(p)
		# l = struct.unpack('i', f_desc_dic[f].read(4))[0]
		# utt = struct.unpack('%df' % l, f_desc_dic[f].read(l * 4))
		#========================================================================2020/04/14 11:26
		try:
			l = struct.unpack('i', f_desc_dic[f].read(4))[0]
			utt = struct.unpack('%df'%l, f_desc_dic[f].read(l * 4))
		except:
			continue
		# ========================================================================
		_nb_samp = len(utt)
		# print('len(utt):{}'.format(_nb_samp))
		#need to verify this part later!!!!!!
		#============================================================================2020/04/13 21:12
		# assert _nb_samp >= nb_samp
		# cut = np.random.randint(low=0, high=_nb_samp - nb_samp)
		# utt = utt[cut:cut + nb_samp]
		# batch.append(utt)
		try:
			# ans存储的说话者编号=['S0002']
			# assert _nb_samp >= nb_samp
			cut = np.random.randint(low = 0, high = _nb_samp - nb_samp)
			utt = utt[cut:cut+nb_samp]
			batch.append(utt)
			ans.append(dic_spk[k.split('/')[1]])
		except:
			num_lines -= 1
			continue
		# ============================================================================

	# return (np.asarray(batch, dtype=np.float32).reshape(len(lines), -1, 1), np.asarray(ans))
	# batch:102->(102,-1,1)
	# ans:102
	#================================================2020/04/18 15:30
	# print("num_lines:{},batch:{},ans:{}".format(num_lines, np.asarray(batch, dtype=np.float32).reshape(num_lines, -1, 1).shape, np.asarray(ans).shape))
	return (np.asarray(batch, dtype=np.float32).reshape(num_lines, -1, 1), np.asarray(ans))

def process_epoch(lines, q, batch_size, nb_samp, dic_spk, base_dir): 
	'''
	Wrapper function for processing mini-batches for the train set once.
	'''
	f_desc_dic = {}
	nb_batch = int(len(lines) / batch_size)
	for i in range(nb_batch):
		while True:
			if q.full():
				sleep(0.1)
			else:
				q.put(compose_batch(lines = lines[i*batch_size: (i+1)*batch_size],
					f_desc_dic = f_desc_dic,
					dic_spk = dic_spk,
					nb_samp = nb_samp,
					base_dir = base_dir))
				break

	for k in f_desc_dic.keys():
		f_desc_dic[k].close()

	return
		

#======================================================================#
#======================================================================#
if __name__ == '__main__':
	#======================================================================#
	#==Yaml load===========================================================#
	#======================================================================#
	_abspath = os.path.abspath(__file__)
	dir_yaml = os.path.splitext(_abspath)[0] + '.yaml'
	# dir_yaml = dir_yaml.replace('/', '\\')
	with open(dir_yaml, 'r') as f_yaml:
		parser = yaml.load(f_yaml)
	
	dir_dev_scp = parser['dev_scp']
	# dir_dev_scp = dir_dev_scp.replace('/', '\\')
	with open(dir_dev_scp, 'r') as f_dev_scp:
		dev_lines = f_dev_scp.readlines()
	dic_spk, list_spk = make_spkdic(dev_lines)
	parser['model']['nb_spk'] = len(list_spk)
	print('# spk: ', len(list_spk))
	parser['model']['batch_size'] = int(parser['batch_size'] / parser['nb_gpu'])
	assert parser['batch_size'] % parser['nb_gpu']  == 0
	
	#select utterances for validation; speaker with 'S'
	# val_lines = []
	# for l in dev_lines:
	# 	if  l[0] == 'd':
	# 		val_lines.append(l)
	dir_val_scp = parser['val_scp']
	# dir_val_scp = dir_val_scp.replace('/', '\\')
	with open(dir_val_scp, 'r') as f_val_scp:
		val_lines = f_val_scp.readlines()

	eval_lines = open(parser['eval_scp'], 'r').readlines()
	trials = open(parser['trials'], 'r').readlines()
	val_trials = open(parser['val_trials'], 'r').readlines()
	nb_batch = int(len(dev_lines) / parser['batch_size'])
	
	global q
	# q中存储的一个单元是(一个batch的语音:(102,-1,1)维 + ans即说话者编号:102维)
	q = queue.Queue(maxsize=1000)
	# dummy_y = np.zeros((parser['batch_size'], 1))

	#======================================================================#
	#==Pre-train===========================================================#
	#======================================================================#
	model, m_name = get_model_pretrn(argDic = parser['model'])
	model_pred = Model(inputs=model.get_layer('input_pretrn').input, outputs=model.get_layer('code_pretrn').output)

	# save_dir = parser['save_dir'] + m_name + '_' + parser['name'] + '/'#==============================2020/04/13 21:01
	save_dir = parser['save_dir'] + parser['name'] + '/'#==============================2020/04/22 11:28
	# save_dir = save_dir.replace('/', '\\')
	#make folders
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	with open(save_dir + 'summary_pretrn.txt' ,'w+') as f_summary:
		model.summary(print_fn=lambda x: f_summary.write(x + '\n'))
	
	f_params = open(save_dir + 'f_params.txt', 'w')
	for k, v in parser.items():
		print(k, v)
		f_params.write('{}:\t{}\n'.format(k, v))
	f_params.write('DNN model params\n')
	
	for k, v in parser['model'].items():
		f_params.write('{}:\t{}\n'.format(k, v))
	print(m_name)
	f_params.write('model_name: %s\n'%m_name)
	f_params.close()

	'''#uncomment to save model architecture in json
	model_json = model.to_json()
	with open(save_dir + 'arc.json', 'w') as f_json:
		f_json.write(model_json)
	'''

	if not os.path.exists(save_dir  + 'results_pretrn/'):
		os.makedirs(save_dir + 'results_pretrn/')
	if not os.path.exists(save_dir  + 'models_pretrn/'):
		os.makedirs(save_dir + 'models_pretrn/')
	f_eer = open(save_dir + 'eers_pretrn.txt', 'w', buffering=1)

	#unzip for model graph visualization (need extra libraries)
	#plot_model(model, to_file=parser['save_dir'] +'visualization.png', show_shapes=True)

	optimizer = eval(parser['optimizer'])(lr=parser['lr'], decay = 0.0, amsgrad = bool(parser['amsgrad']))

	if bool(parser['mg']):
		model_mg = multi_gpu_model(model, gpus=parser['nb_gpu'])
		model_mg.compile(optimizer = optimizer,
			loss = {'s_bs_loss':simple_loss,
					'c_loss':zero_loss},
			loss_weights = {'s_bs_loss':1, 'c_loss':parser['c_lambda']},
			metrics=['accuracy'])
	model.compile(optimizer = optimizer,
			loss = {'s_bs_loss':simple_loss,
					'c_loss':zero_loss},
			loss_weights = {'s_bs_loss':1, 'c_loss': parser['c_lambda']},
		metrics=['accuracy'])
	
	best_val_eer = 99.

	train_times = []  # ===========================================================================2020/05/20 16:30
	total_times = 0  # ===========================================================================2020/05/20 16:30
	for epoch in tqdm(range(parser['epoch'])):
		np.random.shuffle(dev_lines)
		p = Thread(target = process_epoch, args = (dev_lines,
								q,
								parser['batch_size'],
								parser['nb_samp'],
								dic_spk,
								parser['base_dir']))
		p.start()

		#train one epoch!
		loss = 999.
		loss1 = 999.
		loss2 = 999.
		# 记录迭代训练开始时间
		begin_time = time.clock()  # ===========================================================================2020/05/20 16:30
		pbar = tqdm(range(nb_batch))
		for b in pbar:
			pbar.set_description('epoch: %d, loss: %.3f, loss_s+bs: %.3f, loss_c: %.3f'%(epoch, loss, loss1, loss2))
			while True:
				if q.empty():
					# print("q is empty!")#=========================================
					sleep(0.1)

				else:
					# print("q is not empty!")  # =========================================
					x, y = q.get()
					# print("q get x, y")  # =========================================
					# print("x={},y={}".format(x.shape(),y.shape()))
					y = to_categorical(y, num_classes=parser['model']['nb_spk'])
					dummy_y = np.zeros((y.shape[0], 1))# =====================================2020/04/18 15:34
					# print("x={},one-hot编码后 y={}".format(x.shape,y.shape))
					# print("dummy_y={}".format(dummy_y.shape))
					# one-hot编码，要求y为[0,num_classes)间的整型。
					# print("to_categorical")  # =========================================
					if bool(parser['mg']):
						loss, loss1, loss2, acc1, acc2 = model_mg.train_on_batch([x, y], [dummy_y, dummy_y])
					else:
						# print("not mg")  # =========================================
						loss, loss1, loss2, acc1, acc2 = model.train_on_batch([x, y], [dummy_y, dummy_y])
						# loss, loss1, loss2, acc1, acc2 = model.train_on_batch(x, y)#===============================2020/04/18 15:25
					# print("break!")  # =========================================
					break
		p.join()
		# 记录迭代训练结束时间
		train_end_time = time.clock() # ===========================================================================2020/05/20 16:30

		#validate!
		dic_val = compose_spkFeat_dic(lines = val_lines,
							model = model_pred,
							f_desc_dic = {},
							base_dir = parser['base_dir'])
		y = []
		y_score = []
		for smpl in val_trials:
			target, spkMd, utt = smpl.strip().split(' ')
			target = int(target)
			cos_score = cos_sim(dic_val[spkMd], dic_val[utt])
			y.append(target)
			y_score.append(cos_score)
		fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
		eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
		print('\nepoch: %d, val_eer: %f'%(int(epoch), eer))
		f_eer.write('%d %f '%(epoch, eer))

		if float(eer) < best_val_eer:
			best_val_eer = float(eer)
			model.save_weights(save_dir +  'models_pretrn/best_model_on_validation.h5')


		#evaluate!
		dic_eval = compose_spkFeat_dic(lines = eval_lines,
							model = model_pred,
							f_desc_dic = {},
							base_dir = parser['base_dir'])

		f_res = open(save_dir + 'results_pretrn/epoch%s.txt'%(epoch), 'w')
		# f_res = open(save_dir + 'results_pretrn\epoch%s.txt' % (epoch), 'w')
		y = []
		y_score = []
		for smpl in trials:
			target, spkMd, utt = smpl.strip().split(' ')
			target = int(target)
			cos_score = cos_sim(dic_eval[spkMd], dic_eval[utt])
			y.append(target)
			y_score.append(cos_score)
			f_res.write('{score} {target}\n'.format(score=cos_score,target=target))
		f_res.close()
		fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
		'''
		#prints threshold
		#thresh = interp1d(fpr, thresholds)(eer)
		print(thresh)
		'''
		eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
		print('\nepoch: %d, eer: %f'%(int(epoch), eer))
		f_eer.write('%f\n'%(eer))

		if not bool(parser['save_best_only']):
			model.save_weights(save_dir +  'models_pretrn/%d-%.4f.h5'%(epoch, eer))

		end_time = time.clock()  # ===========================================================================2020/05/20 16:30
		total_times += train_end_time - begin_time  # ===========================================================================2020/05/20 16:30
		train_times.append(str(begin_time)+'_'+str(train_end_time)+'_'+str(end_time)+'_'+str(train_end_time-begin_time))  # ===========================================================================2020/05/20 16:30
		print("epoch:{},耗时:{}s".format(epoch, str(train_end_time - begin_time)))  # ===========================================================================2020/05/20 16:30
	f_eer.close()
	# ===========================================================================2020/05/20 16:30
	# 将时间写入文件
	with open('RawNet_PreTrains_epoch{}_spk90_UttPerSpk5迭代耗时.txt'.format(parser['epoch']), mode='w',encoding='utf-8') as wf:
		wf.write("epoch{}_平均每次训练耗时:{}\n".format(parser['epoch'], total_times / parser['epoch']))
		wf.write("开始训练时间_结束训练时间_结束epoch时间(包括验证读写文件等)_耗时(结束训练时间-开始训练时间)\n")
		for line in train_times:
			wf.write(line + '\n')
	# ===========================================================================结束
	#======================================================================#
	#==Train RawNet========================================================#
	#======================================================================#
	model, m_name = get_model(argDic = parser['model'])
	model_pred = Model(inputs=model.get_layer('input_RawNet').input, outputs=model.get_layer('code_RawNet').output)
	model.load_weights(save_dir+'models_pretrn/best_model_on_validation.h5', by_name = True)


	with open(save_dir + 'summary_RawNet.txt' ,'w+') as f_summary:
		model.summary(print_fn=lambda x: f_summary.write(x + '\n'))
	
	if not os.path.exists(save_dir  + 'results_RawNet/'):
		os.makedirs(save_dir + 'results_RawNet/')
	if not os.path.exists(save_dir  + 'models_RawNet/'):
		os.makedirs(save_dir + 'models_RawNet/')
	f_eer = open(save_dir + 'eers_RawNet.txt', 'w', buffering=1)

	optimizer = eval(parser['optimizer'])(lr=parser['lr'], decay = parser['opt_decay'], amsgrad = bool(parser['amsgrad']))
	parser['c_lambda'] = parser['c_lambda'] * 0.01
	if bool(parser['mg']):
		model_mg = multi_gpu_model(model, gpus=parser['nb_gpu'])
		model_mg.compile(optimizer = optimizer,
			loss = {'gru_s_bs_loss':simple_loss,
					'gru_c_loss':zero_loss},
			loss_weights = {'gru_s_bs_loss':1, 'gru_c_loss':parser['c_lambda']},
			metrics=['accuracy'])
	model.compile(optimizer = optimizer,
			loss = {'gru_s_bs_loss':simple_loss,
					'gru_c_loss':zero_loss},
			loss_weights = {'gru_s_bs_loss':1, 'gru_c_loss': parser['c_lambda']},
		metrics=['accuracy'])
	
	best_val_eer = 99.
	train_times = []  # ===========================================================================2020/05/20 16:30
	total_times = 0  # ===========================================================================2020/05/20 16:30
	for epoch in tqdm(range(parser['epoch'])):
		np.random.shuffle(dev_lines)
		p = Thread(target = process_epoch, args = (dev_lines,
								q,
								parser['batch_size'],
								parser['nb_samp'],
								dic_spk,
								parser['base_dir']))
		p.start()

		#train one epoch!
		loss = 999.
		loss1 = 999.
		loss2 = 999.
		# 记录迭代训练开始时间
		begin_time = time.clock()  # ===========================================================================2020/05/20 16:30s
		pbar = tqdm(range(nb_batch))
		for b in pbar:
			pbar.set_description('epoch: %d, loss: %.3f, loss_s+bs: %.3f, loss_c: %.3f'%(epoch, loss, loss1, loss2))
			while True:
				if q.empty():
					sleep(0.1)
	
				else:
					x, y = q.get()
					y = to_categorical(y, num_classes=parser['model']['nb_spk'])
					# y = to_categorical(y, num_classes=parser['model']['nb_spk'])# =====================================2020/04/27 22:25
					dummy_y = np.zeros((y.shape[0], 1))  # =====================================2020/04/27 22:25
					if bool(parser['mg']):
						loss, loss1, loss2, acc1, acc2 = model_mg.train_on_batch([x, y], [dummy_y, dummy_y])
					else:
						loss, loss1, loss2, acc1, acc2 = model.train_on_batch([x, y], [dummy_y, dummy_y])
					
					break
		p.join()
		# 记录迭代训练结束时间
		train_end_time = time.clock()  # ===========================================================================2020/05/20 16:30
		#validate!
		dic_val = compose_spkFeat_dic(lines = val_lines,
							model = model_pred,
							f_desc_dic = {},
							base_dir = parser['base_dir'])
		y = []
		y_score = []
		for smpl in val_trials:
			target, spkMd, utt = smpl.strip().split(' ')
			target = int(target)
			cos_score = cos_sim(dic_val[spkMd], dic_val[utt])
			y.append(target)
			y_score.append(cos_score)
		fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
		eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
		print('\nepoch: %d, val_eer: %f'%(int(epoch), eer))
		f_eer.write('%d %f '%(epoch, eer))
		
		if float(eer) < best_val_eer:
			best_val_eer = float(eer)
			model.save_weights(save_dir +  'models_RawNet/best_model_on_validation.h5')
	
	
		#evaluate!
		dic_eval = compose_spkFeat_dic(lines = eval_lines,
							model = model_pred,
							f_desc_dic = {},
							base_dir = parser['base_dir'])
	
		f_res = open(save_dir + 'results_RawNet/epoch%s.txt'%(epoch), 'w')
		# f_res = open(save_dir + 'results_RawNet\epoch%s.txt' % (epoch), 'w')
		y = []
		y_score = []
		for smpl in trials:
			target, spkMd, utt = smpl.strip().split(' ')
			target = int(target)
			cos_score = cos_sim(dic_eval[spkMd], dic_eval[utt])
			y.append(target)
			y_score.append(cos_score)
			f_res.write('{score} {target}\n'.format(score=cos_score,target=target))
		f_res.close()
		fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
		'''
		#prints threshold
		#thresh = interp1d(fpr, thresholds)(eer)
		print(thresh)
		'''
		eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
		print('\nepoch: %d, eer: %f'%(int(epoch), eer))
		f_eer.write('%f\n'%(eer))

		if not bool(parser['save_best_only']):
			model.save_weights(save_dir +  'models_RawNet/%d-%.4f.h5'%(epoch, eer))
		end_time = time.clock()  # ===========================================================================2020/05/20 16:30
		total_times += train_end_time - begin_time  # ===========================================================================2020/05/20 16:30
		train_times.append(str(begin_time) + '_' + str(train_end_time) + '_' + str(end_time) + '_' + str(train_end_time - begin_time))  # ===========================================================================2020/05/20 16:30
		print("epoch:{},耗时:{}s".format(epoch, str(train_end_time - begin_time)))  # ===========================================================================2020/05/20 16:30
	f_eer.close()
	# ===========================================================================2020/05/20 16:30
	# 将时间写入文件
	with open('RawNet_RawNet_epoch{}_spk90_UttPerSpk5迭代耗时.txt'.format(parser['epoch']), mode='w', encoding='utf-8') as wf:
		wf.write("epoch{}_平均每次训练耗时:{}\n".format(parser['epoch'], total_times / parser['epoch']))
		wf.write("开始训练时间_结束训练时间_结束epoch时间(包括验证读写文件等)_耗时(结束训练时间-开始训练时间)\n")
		for line in train_times:
			wf.write(line + '\n')
	# ===========================================================================结束
	#======================================================================#
	#==Extract RawNet Embeddings===========================================#
	#======================================================================#

	model.load_weights(save_dir + 'models_RawNet/best_model_on_validation.h5')
	if not os.path.exists(parser['gru_embeddings']):
		os.makedirs(parser['gru_embeddings'])

	print('Extracting Embeddings from GRU model: dev set')
	dev_dic_embeddings = compose_spkFeat_dic(lines = dev_lines,
		model = model_pred,
		f_desc_dic = {},
		base_dir = parser['base_dir'])

	print('Extracting Embeddings from GRU model: eval set')
	eval_dic_embeddings = compose_spkFeat_dic(lines = eval_lines,
		model = model_pred,
		f_desc_dic = {},
		base_dir = parser['base_dir'])
	# =====================================================================================2020/05/18 11:16
	print('Extracting Embeddings from GRU model: val set')
	val_dic_embeddings = compose_spkFeat_dic(lines=val_lines,
											 model=model_pred,
											 f_desc_dic={},
											 base_dir=parser['base_dir'])
	# =====================================================================================结束

	f_embeddings = open(parser['gru_embeddings'] + 'speaker_embeddings_RawNet', 'wb')
	# =====================================================================================2020/05/18 11:16
	pk.dump({'dev_dic_embeddings': dev_dic_embeddings, 'eval_dic_embeddings': eval_dic_embeddings,
			 'val_dic_embeddings': val_dic_embeddings},
			f_embeddings,
			protocol=pk.HIGHEST_PROTOCOL)
	# pk.dump({'dev_dic_embeddings': dev_dic_embeddings, 'eval_dic_embeddings': eval_dic_embeddings},
	#         f_embeddings,
	#         protocol=pk.HIGHEST_PROTOCOL)
	# =====================================================================================结束



