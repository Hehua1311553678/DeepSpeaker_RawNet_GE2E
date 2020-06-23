import numpy as np
import struct
import os
import soundfile as sf
from multiprocessing import Process

def pre_emp(x):
	'''
	Pre-emphasize raw waveform '''
	return np.asarray(x[1:] - 0.97 * x[:-1], dtype=np.float32)

def extract_waveforms(lines, dir_trg):

	f_scp_pe = open(dir_trg + '_pe.scp', 'w')
	f_ark_pe = open(dir_trg + '_pe.ark', 'wb')
	for line in lines:
		key, fn = line.strip().split(' ')
		print(key, fn)

		wav ,_= sf.read(fn, dtype='int16')
		wav = pre_emp(wav)
		pointer = f_ark_pe.tell()
		arkf_dir = os.path.abspath(dir_trg + '_pe.ark').replace('\\', '/')
		# arkf_dir = '/'.join(arkf_dir.split('/')[-4:])
		arkf_dir = '/'.join(arkf_dir.split('/')[-3:])#=========================================2020/04/27 23:28
		f_scp_pe.write('%s %s %d\n'%(key, arkf_dir, pointer))
		f_ark_pe.write(struct.pack('<i', wav.shape[0]))
		f_ark_pe.write(struct.pack('<%df'%wav.shape[0], *wav.tolist()))

	f_scp_pe.close()
	f_ark_pe.close()
	
def join_scp(f_dir, nb_proc):
	f_scp_pe = open(f_dir + '_pe.scp', 'w')
	for i in range(nb_proc):
		with open(f_dir + '_%d_pe.scp'%i, 'r') as f_read:
			lines = f_read.readlines()
		for line in lines:
			f_scp_pe.write(line)
		os.remove(f_dir + '_%d_pe.scp'%i)
	f_scp_pe.close()


'''初始模板'''
# DB_dir = '/DB/VoxCeleb1/voxceleb1_wav/'#directory where downloaded VoxCeleb1 dataset exists
# scp_dir = '/DB/'#directory to store processed raw waveforms
# dataset = 'dev'#execute this script with either 'dev' or 'eval'

'''VoxCeleb1'''
# DB_dir = '/hehua/DB/VoxCeleb1/voxceleb1_wav/'#directory where downloaded VoxCeleb1 dataset exists
# scp_dir = '/hehua/DB/'#directory to store processed raw waveforms
# dataset = 'dev'#execute this script with either 'dev' or 'eval'

'''aishell1'''
# DB_dir = 'F:\hehua\DB\Aishell1\\aishell-1\\test'#directory where downloaded VoxCeleb1 dataset exists
# scp_dir = 'F:\hehua\DB\Aishell1\scp\\test\\'#directory to store processed raw waveforms
# dataset = 'test'#execute this script with either 'dev' or 'test' or 'train'
# DB_dir = 'D:\DB\Aishell1\\aishell-1\\train'#directory where downloaded VoxCeleb1 dataset exists
# scp_dir = 'D:\Codes\Python\graduatedesign\RawNet\RawNet-master\keras\\aishell1\scp\\train\\'#directory to store processed raw waveforms
# dataset = 'train'#execute this script with either 'dev' or 'test' or 'train'

'''vctk'''#===================================================2020/04/21 23:24
# DB_dir = 'E:\DB\VCTK\VCTK-Corpus\\train'#directory where downloaded VoxCeleb1 dataset exists
# scp_dir = 'scp\\train\\'#directory to store processed raw waveforms
# dataset = 'train'#execute this script with either 'dev' or 'test' or 'train'
#===================================================2020/04/25 08:25
# DB_dir = '/home/typ/hehua/DB/VCTK/VCTK-Corpus/train'#directory where downloaded VoxCeleb1 dataset exists
DB_dir = '/hehua/DB/VCTK/VCTK-Corpus/test'#directory where downloaded VoxCeleb1 dataset exists
# DB_dir = '/home/typ/hehua/DB/VCTK_rawnet_test/train'#directory where downloaded VoxCeleb1 dataset exists
scp_dir = 'scp/test/'#directory to store processed raw waveforms
dataset = 'test'#execute this script with either 'dev' or 'test' or 'train'

if __name__ == '__main__':
	nb_proc = 12
	if not os.path.exists(scp_dir):
		os.makedirs(scp_dir)

	list_f_dir = []
	for r, ds, fs in os.walk(DB_dir):
		for f in fs:
			fn = '/'.join([r, f]).replace('\\', '/')
			key = '/'.join(fn.split('/')[-3:])#==============================================#
			# key = fn.split('/')[-1]#==============================================2020/04/21 23:31


			#==============================================================2020/04/21 23:32
			# if dataset=='vctk':
			# 	list_f_dir.append('%s %s\n' % (key, fn))
			if dataset == 'dev':
				if key[0] == 'd':
					list_f_dir.append('%s %s\n'%(key, fn))

			elif dataset == 'test':
				if key[2] == 's':
					list_f_dir.append('%s %s\n'%(key, fn))
			elif dataset == 'train':
				if key[2] == 'a':
					list_f_dir.append('%s %s\n'%(key, fn))
			else:
				raise ValueError('sub-dataset not known!! use either "dev" or "eval" for "dataset".')
			# ==============================================================
			
	print('='*5 + 'done' + '='*5)
	print(len(list_f_dir))


	list_proc = []
	nb_utt_per_proc = int(len(list_f_dir) / nb_proc)
	for i in range(nb_proc):
		if i == nb_proc - 1:
			lines = list_f_dir[i * nb_utt_per_proc : ]
		else:
			lines = list_f_dir[i * nb_utt_per_proc : (i+1) * nb_utt_per_proc]

		print(len(lines))
		list_proc.append(Process(target = extract_waveforms, args = (lines, scp_dir+'%s_wav_%d'%(dataset, i))))
		print('%d'%i)

	for i in range(nb_proc):
		list_proc[i].start()
		print('start %d'%i)
	for i in range(nb_proc):
		list_proc[i].join()

	join_scp(f_dir = scp_dir + '%s_wav'%dataset, nb_proc = nb_proc)
