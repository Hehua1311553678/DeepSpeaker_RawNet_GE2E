
# train-clean-100: 251 speaker, 28539 utterance
# train-clean-360: 921 speaker, 104104 utterance
# test-clean: 40 speaker, 2620 utterance
# merged test: 80 speaker, 5323 utterance
# batchisize 32*3 : train on triplet: 5s - > 3.1s/steps , softmax pre train: 3.1 s/steps


import logging
from time import time

import numpy as np
import sys
import os
import random
from keras.optimizers import Adam
from keras.layers.core import Dense
from keras.models import Model

# import constants as c
import constants_vctk as c#=================================================================================================2020/04/17 20:49
import select_batch
from pre_process import data_catalog, preprocess_and_save
from models import convolutional_model, convolutional_model_simple, recurrent_model
from random_batch import stochastic_mini_batch
from triplet_loss import deep_speaker_loss
from utils import get_last_checkpoint_if_any, create_dir_and_delete_content
from test_model import eval_model


def create_dict(files,labels,spk_uniq):
    train_dict = {}
    for i in range(len(spk_uniq)):
        train_dict[spk_uniq[i]] = []

    for i in range(len(labels)):
        train_dict[labels[i]].append(files[i])
    # ================================================================================2020/05/21 11：51
    # 修改说话人个数
    train_dict_fixed_spk_num = {}
    spk_uniq = spk_uniq[:c.Spk_num]
    for spk in spk_uniq:
        train_dict_fixed_spk_num[spk] = train_dict[spk]
    train_dict = train_dict_fixed_spk_num
    # 修改每人音频数
    for spk in spk_uniq:
        train_dict[spk] = train_dict[spk][:c.UttPerSpk]
    # ================================================================================结束
    for spk in spk_uniq:
        if len(train_dict[spk]) < 2:
            train_dict.pop(spk)
    unique_speakers=list(train_dict.keys())
    return train_dict, unique_speakers

def main(libri_dir=c.DATASET_DIR):

    PRE_TRAIN = c.PRE_TRAIN
    logging.info('Looking for fbank features [.npy] files in {}.'.format(libri_dir))
    libri = data_catalog(libri_dir)

    if len(libri) == 0:
        logging.warning('Cannot find npy files, we will load audio, extract features and save it as npy file')
        logging.warning('Waiting for preprocess...')
        preprocess_and_save(c.WAV_DIR, c.DATASET_DIR)
        libri = data_catalog(libri_dir)
        if len(libri) == 0:
            logging.warning('Have you converted flac files to wav? If not, run audio/convert_flac_2_wav.sh')
            exit(1)
    unique_speakers = libri['speaker_id'].unique()
    spk_utt_dict, unique_speakers = create_dict(libri['filename'].values,libri['speaker_id'].values,unique_speakers)
    select_batch.create_data_producer(unique_speakers, spk_utt_dict)

    batch = stochastic_mini_batch(libri, batch_size=c.BATCH_SIZE, unique_speakers=unique_speakers)
    batch_size = c.BATCH_SIZE * c.TRIPLET_PER_BATCH
    x, y = batch.to_inputs()
    b = x[0]
    num_frames = b.shape[0]
    train_batch_size = batch_size
    #batch_shape = [batch_size * num_frames] + list(b.shape[1:])  # A triplet has 3 parts.
    input_shape = (num_frames, b.shape[1], b.shape[2])

    logging.info('num_frames = {}'.format(num_frames))
    logging.info('batch size: {}'.format(batch_size))
    logging.info('input shape: {}'.format(input_shape))
    logging.info('x.shape : {}'.format(x.shape))
    # 按理x.shape:(batchsize, num_frames, 64, 1)
    orig_time = time()
    model = convolutional_model(input_shape=input_shape, batch_size=batch_size, num_frames=num_frames)
    logging.info(model.summary())
    gru_model = None
    if c.COMBINE_MODEL:
        gru_model = recurrent_model(input_shape=input_shape, batch_size=batch_size, num_frames=num_frames)
        logging.info(gru_model.summary())
    grad_steps = 0

    if PRE_TRAIN:
        last_checkpoint = get_last_checkpoint_if_any(c.PRE_CHECKPOINT_FOLDER)
        if last_checkpoint is not None:
            logging.info('Found pre-training checkpoint [{}]. Resume from here...'.format(last_checkpoint))
            x = model.output
            x = Dense(len(unique_speakers), activation='softmax', name='softmax_layer')(x)
            pre_model = Model(model.input, x)
            pre_model.load_weights(last_checkpoint)
            grad_steps = int(last_checkpoint.split('_')[-2])
            logging.info('Successfully loaded pre-training model')

    else:
        last_checkpoint = get_last_checkpoint_if_any(c.CHECKPOINT_FOLDER)
        if last_checkpoint is not None:
            logging.info('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
            model.load_weights(last_checkpoint)
            grad_steps = int(last_checkpoint.split('_')[-2])
            logging.info('[DONE]')
        if c.COMBINE_MODEL:
            last_checkpoint = get_last_checkpoint_if_any(c.GRU_CHECKPOINT_FOLDER)
            if last_checkpoint is not None:
                logging.info('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
                gru_model.load_weights(last_checkpoint)
                logging.info('[DONE]')

    #adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer='adam', loss=deep_speaker_loss)
    if c.COMBINE_MODEL:
        gru_model.compile(optimizer='adam', loss=deep_speaker_loss)
    print("model_build_time",time()-orig_time)
    logging.info('Starting training...')
    lasteer = 10
    eer = 1
    # ======================================================================2020/05/21 10：38
    train_times = []  # ===========================================================================2020/05/20 16:30
    total_times = 0  # ===========================================================================2020/05/20 16:30
    # 迭代10个epoch，每个epoch200个batch
    # while True:
    os.makedirs(c.BEST_CHECKPOINT_FOLDER, exist_ok=True)
    while grad_steps < 2001 :
        # ======================================================================结束
        orig_time = time()
        x, _ = select_batch.best_batch(model, batch_size=c.BATCH_SIZE)
        print("select_batch_time:", time() - orig_time)
        y = np.random.uniform(size=(x.shape[0], 1))
        # If "ValueError: Error when checking target: expected ln to have shape (None, 512) but got array with shape (96, 1)"
        # please modify line 121 to following line
        # y = np.random.uniform(size=(x.shape[0], 512))
        logging.info('== Presenting step #{0}'.format(grad_steps))
        orig_time = time()
        # ======================================================================2020/05/21 10：38
        # 记录训练batch时间
        # 记录迭代训练开始时间
        begin_time = time()  # ===========================================================================2020/05/20 16:30
        loss = model.train_on_batch(x, y)
        # 记录迭代训练结束时间
        train_end_time = time()  # ===========================================================================2020/05/20 16:30
        # ======================================================================结束
        logging.info('== Processed in {0:.2f}s by the network, training loss = {1}.'.format(time() - orig_time, loss))
        if c.COMBINE_MODEL:
            loss1 = gru_model.train_on_batch(x, y)
            logging.info( '== Processed in {0:.2f}s by the gru-network, training loss = {1}.'.format(time() - orig_time, loss1))
            with open(c.GRU_CHECKPOINT_FOLDER + '/losses_gru.txt', "a") as f:
                f.write("{0},{1}\n".format(grad_steps, loss1))
        # record training loss
        with open(c.LOSS_LOG, "a") as f:
            f.write("{0},{1}\n".format(grad_steps, loss))
        if (grad_steps) % 10 == 0:
            fm1, tpr1, acc1, eer1 = eval_model(model, train_batch_size, test_dir=c.DATASET_DIR, check_partial=True, gru_model=gru_model)
            logging.info('test training data EER = {0:.3f}, F-measure = {1:.3f}, Accuracy = {2:.3f} '.format(eer1, fm1, acc1))
            with open(c.CHECKPOINT_FOLDER + '/train_acc_eer.txt', "a") as f:
                f.write("{0},{1},{2},{3}\n".format(grad_steps, eer1, fm1, acc1))

        if (grad_steps ) % c.TEST_PER_EPOCHS == 0 :
            fm, tpr, acc, eer = eval_model(model,train_batch_size, test_dir=c.TEST_DIR,gru_model=gru_model)
            logging.info('== Testing model after batch #{0}'.format(grad_steps))
            logging.info('EER = {0:.3f}, F-measure = {1:.3f}, Accuracy = {2:.3f} '.format(eer, fm, acc))
            with open(c.TEST_LOG, "a") as f:
                f.write("{0},{1},{2},{3}\n".format(grad_steps, eer, fm, acc))

        # checkpoints are really heavy so let's just keep the last one.
        if (grad_steps ) % c.SAVE_PER_EPOCHS == 0:
            create_dir_and_delete_content(c.CHECKPOINT_FOLDER)
            model.save_weights('{0}/model_{1}_{2:.5f}.h5'.format(c.CHECKPOINT_FOLDER, grad_steps, loss))
            if c.COMBINE_MODEL:
                gru_model.save_weights('{0}/grumodel_{1}_{2:.5f}.h5'.format(c.GRU_CHECKPOINT_FOLDER, grad_steps, loss1))
            if eer < lasteer:
                files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".h5"),
                                      map(lambda f: os.path.join(c.BEST_CHECKPOINT_FOLDER, f), os.listdir(c.BEST_CHECKPOINT_FOLDER))),
                               key=lambda file: file.split('/')[-1].split('.')[-2], reverse=True)
                lasteer = eer
                for file in files[:-4]:
                    logging.info("removing old model: {}".format(file))
                    os.remove(file)
                model.save_weights(c.BEST_CHECKPOINT_FOLDER+'/best_model{0}_{1:.5f}.h5'.format(grad_steps, eer))
                if c.COMBINE_MODEL:
                    files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".h5"),
                                          map(lambda f: os.path.join(c.BEST_CHECKPOINT_FOLDER, f),
                                              os.listdir(c.BEST_CHECKPOINT_FOLDER))),
                                   key=lambda file: file.split('/')[-1].split('.')[-2], reverse=True)
                    lasteer = eer
                    for file in files[:-4]:
                        logging.info("removing old model: {}".format(file))
                        os.remove(file)
                    gru_model.save_weights(c.BEST_CHECKPOINT_FOLDER+'/best_gru_model{0}_{1:.5f}.h5'.format(grad_steps, eer))

        grad_steps += 1
        end_time = time()  # ===========================================================================2020/05/20 16:30
        total_times += train_end_time - begin_time  # ===========================================================================2020/05/20 16:30
        train_times.append(str(begin_time) + '_' + str(train_end_time) + '_' + str(end_time) + '_' + str(train_end_time - begin_time))  # ===========================================================================2020/05/20 16:30
        print("步数:{},耗时:{}s".format(grad_steps, str(train_end_time - begin_time)))  # ===========================================================================2020/05/20 16:30
    # ===========================================================================2020/05/20 16:30
    # 将时间写入文件
    with open('DeepSpeaker_epoch10_spk{}_suttperspk{}_迭代耗时.txt'.format(str(c.Spk_num), str(c.UttPerSpk)), mode='w', encoding='utf-8') as wf:
        wf.write("步数{}_平均每次训练耗时:{}\n".format(grad_steps, total_times / grad_steps))
        wf.write("开始训练时间_结束训练时间_结束步数训练时间(包括验证读写文件等)_耗时(结束训练时间-开始训练时间)\n")
        for line in train_times:
            wf.write(line + '\n')
    # ===========================================================================结束



if __name__ == '__main__':
    logging.basicConfig(handlers=[logging.StreamHandler(stream=sys.stdout)], level=logging.INFO,
                        format='%(asctime)-15s [%(levelname)s] %(filename)s/%(funcName)s | %(message)s')
    main()
