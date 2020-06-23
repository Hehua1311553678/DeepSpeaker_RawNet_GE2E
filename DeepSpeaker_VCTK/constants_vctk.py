Spk_num = 90#==============================================================2020/05/21 11:30
UttPerSpk = 5#==============================================================2020/05/21 11:30

# DATASET_DIR = 'audio/LibriSpeechSamples/train-clean-100-npy/'
DATASET_DIR = 'audio/vctk-npy/'#==============================================================2020/04/02 23:00
TEST_DIR = 'audio/vctk-npy/'#==============================================================2020/04/02 23:00
WAV_DIR = 'audio/vctk-npy/'#==============================================================2020/04/02 23:00
KALDI_DIR = ''

BATCH_SIZE = 32        #must be even
TRIPLET_PER_BATCH = 3

SAVE_PER_EPOCHS = 200
TEST_PER_EPOCHS = 200
CANDIDATES_PER_BATCH = 640       # 18s per batch
TEST_NEGATIVE_No = 99


NUM_FRAMES = 160   # 299 - 16*2
SAMPLE_RATE = 16000
TRUNCATE_SOUND_SECONDS = (0.2, 1.81)  # (start_sec, end_sec)
ALPHA = 0.2
HIST_TABLE_SIZE = 10
NUM_SPEAKERS = 251
DATA_STACK_SIZE = 10

CHECKPOINT_FOLDER = 'vctk_spk'+str(Spk_num)+'_UttPerSpk'+str(UttPerSpk)+'/checkpoints'
BEST_CHECKPOINT_FOLDER = 'vctk_spk'+str(Spk_num)+'_UttPerSpk'+str(UttPerSpk)+'/best_checkpoint'
PRE_CHECKPOINT_FOLDER = 'vctk_spk'+str(Spk_num)+'_UttPerSpk'+str(UttPerSpk)+'/pretraining_checkpoints'
GRU_CHECKPOINT_FOLDER = 'vctk_spk'+str(Spk_num)+'_UttPerSpk'+str(UttPerSpk)+'/gru_checkpoints'

LOSS_LOG= CHECKPOINT_FOLDER + '/losses.txt'
TEST_LOG= CHECKPOINT_FOLDER + '/acc_eer.txt'

PRE_TRAIN = False

COMBINE_MODEL = False
