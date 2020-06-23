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
from time import sleep

from keras.utils import multi_gpu_model, plot_model, to_categorical
from keras.optimizers import *
from keras.models import Model
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from model_RawNet import get_model

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


if __name__ == '__main__':
    # ======================================================================#
    # ==Yaml load===========================================================#
    # ======================================================================#
    dir_yaml = '01-trn_RawNet.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser = yaml.load(f_yaml)

    dir_dev_scp = parser['dev_scp']
    # dir_dev_scp = dir_dev_scp.replace('/', '\\')
    with open(dir_dev_scp, 'r') as f_dev_scp:
        dev_lines = f_dev_scp.readlines()
    dic_spk, list_spk = make_spkdic(dev_lines)
    parser['model']['nb_spk'] = len(list_spk)

    val_lines = open(parser['val_scp'], 'r').readlines()
    save_dir = parser['save_dir'] + parser['name'] + '/'  # ==============================2020/04/22 11:28

    # =============================================读取原来的特征pickle
    _ = pk.load(open(parser['gru_embeddings'] + 'speaker_embeddings_RawNet', 'rb'))
    dev_dic_embeddings = _['dev_dic_embeddings']
    eval_dic_embeddings = _['eval_dic_embeddings']
    # =============================================

    # ======================================================================#
    # ==Extract RawNet Embeddings===========================================#
    # ======================================================================#
    model, m_name = get_model(argDic=parser['model'])
    model_pred = Model(inputs=model.get_layer('input_RawNet').input, outputs=model.get_layer('code_RawNet').output)
    model.load_weights(save_dir + 'models_RawNet/best_model_on_validation.h5')

    if not os.path.exists(parser['gru_embeddings']):
        os.makedirs(parser['gru_embeddings'])

    print('Extracting Embeddings from GRU model: eval set')
    val_dic_embeddings = compose_spkFeat_dic(lines=val_lines,
                                              model=model_pred,
                                              f_desc_dic={},
                                              base_dir=parser['base_dir'])

    f_embeddings = open(parser['gru_embeddings'] + 'speaker_embeddings_RawNet1', 'wb')
    pk.dump({'dev_dic_embeddings': dev_dic_embeddings, 'eval_dic_embeddings': eval_dic_embeddings, 'val_dic_embeddings':val_dic_embeddings},
            f_embeddings,
            protocol=pk.HIGHEST_PROTOCOL)
