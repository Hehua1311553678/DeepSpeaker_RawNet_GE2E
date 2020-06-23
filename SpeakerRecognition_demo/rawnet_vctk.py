import pickle as pk
import yaml
import os
import numpy as np
import struct

from keras.models import Model
import soundfile as sf

from model_RawNet import get_model as get_model_RawNet
from model_bvec import get_model as get_model_bvec

# ==========================================================================动态分配显存
# import tensorflow as tf
# import keras
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
# ==========================================================================
# ==========================================================================
import tensorflow as tf
import keras  #引入keras
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)
keras.backend.clear_session()
# ==========================================================================

# def get_gl_mean_std(path_embeddings = 'G:\hehua\RawNet\\vctk_mini\exp\data\speaker_embeddings_RawNet'):
def get_gl_mean_std(path_embeddings = 'data/speaker_embeddings_RawNet'):
    '''提取训练集嵌入均值方差'''
    _ = pk.load(open(path_embeddings, 'rb'))
    dev_dic_embeddings = _['dev_dic_embeddings']
    dev_list_embeddings = []
    for k in dev_dic_embeddings.keys():
        dev_list_embeddings.append(dev_dic_embeddings[k])
    gl_mean = np.mean(dev_list_embeddings, axis=0)
    gl_std = np.std(dev_list_embeddings, axis=0)
    return gl_mean, gl_std

def pre_emp(x):
	'''
	Pre-emphasize raw waveform '''
	return np.asarray(x[1:] - 0.97 * x[:-1], dtype=np.float32)

def extract_waveforms(wav_path):
    '''提取声学特征
    wav_path:音频路径;
    '''
    wav, _ = sf.read(wav_path, dtype='int16')
    wav = pre_emp(wav)
    return wav.tolist()

def get_embeddings(wav_feature, models_dir = 'models/reproduce_RawNet/', embeddings_dir = 'embeddings/'):
    '''提取说话人嵌入
    wav_feature:音频声学特征；
    models_dir:模型存储根路径;
    embeddings_dir:嵌入存储路径;
    '''
    dir_yaml = '01-trn_RawNet.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser = yaml.load(f_yaml)
    model, m_name = get_model_RawNet(argDic=parser['model'])
    model_pred = Model(inputs=model.get_layer('input_RawNet').input, outputs=model.get_layer('code_RawNet').output)
    model.load_weights(models_dir + 'models_RawNet/best_model_on_validation.h5')

    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    utt = np.asarray(wav_feature, dtype=np.float32)
    spkFeat = model_pred.predict(utt.reshape(1, -1, 1))[0]  # extract speaker embedding from utt

    return spkFeat

def get_score(embedding1, embedding2, models_dir = 'models/reproduce_bvec/'):
    '''后端网络打分
    embedding1:第一条音频嵌入特征;
    embedding2:第二条音频嵌入特征;
    '''
    dir_yaml = '02-trn_bvec.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser = yaml.load(f_yaml)
    model, m_name = get_model_bvec(argDic=parser['model'])
    model.load_weights(models_dir + 'models_bvec/best_model_on_validation.h5')
    score = model.predict([np.asarray([embedding1]), np.asarray([embedding2])])[:1]
    return score

def sv_demo(wav_paths):
    gl_mean, gl_std = get_gl_mean_std()

    wav_path1 = wav_paths[0]
    wav_path2 = wav_paths[1]
    # 提取声学特征
    wav_feature1 = extract_waveforms(wav_path1)
    wav_feature2 = extract_waveforms(wav_path2)

    # 提取embedding特征
    embdeding1 = get_embeddings(wav_feature1)
    embdeding2 = get_embeddings(wav_feature2)

    embdeding1 = (embdeding1 - gl_mean) / gl_std
    embdeding2 = (embdeding2 - gl_mean) / gl_std
    # 打分
    score = get_score(embdeding1, embdeding2)[:, 1]

    print("score:{}".format(score))
    return score[0]

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 如有GPU的话取消注释该行，GPU会加速特征提取.
    wav_paths = ['audios/p339_001.wav', 'audios/p339_002.wav', 'audios/p340_001.wav']
    # wav_paths = ['audios/p362_001.wav', 'audios/p362_002.wav', 'audios/p340_001.wav']
    sv_demo(wav_paths)

