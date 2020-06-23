import os

# paths_scp_before = ['scp/train/原来train_wav_pe.scp', 'scp/test/原来test_wav_pe.scp', 'scp/dev/原来dev_wav_pe.scp']
paths_scp_before = ['scp/train/train_wav_pe.scp', 'scp/test/test_wav_pe.scp', 'scp/dev/dev_wav_pe.scp']
# audio_nums = [5, 10, 20, 50, 100]
audio_nums = [5]
spk_nums = [5, 10, 20, 50, 90]
# audio_nums = [1]

for spk_num_per_epoch in spk_nums:
    for audio_num_per_person in audio_nums:
        for path_scp_before in paths_scp_before:
            path_scp_behind = os.path.splitext(path_scp_before)[0] + '_spk' + str(spk_num_per_epoch) \
                              + '_UttPerSpk' + str(audio_num_per_person) + '.scp'
            # 说话者：语音记录
            lines = open(path_scp_before, mode='r', encoding='utf-8').readlines()
            dict_spk_utt = {}
            for line in lines:
                spk = line.split('/')[1]
                if spk not in dict_spk_utt.keys():
                    dict_spk_utt[spk] = []
                dict_spk_utt[spk].append(line)
            # 只要spk_num_per_epoch个说话人
            index = 0
            dict_spk_utt_behind = {}
            for key in dict_spk_utt.keys():
                if index >= spk_num_per_epoch:
                    break
                dict_spk_utt_behind[key] = dict_spk_utt[key]
                index += 1
            dict_spk_utt = dict_spk_utt_behind
            # 每个说话人只要audio_num_per_person个语音记录
            list_spk_utt_changed = []
            for spk in dict_spk_utt.keys():
                list_spk_utt_changed += dict_spk_utt[spk][:audio_num_per_person]
            # 调整后的语音记录存入文件
            wf = open(path_scp_behind, mode='w', encoding='utf-8')
            for line in list_spk_utt_changed:
                wf.write(line)

