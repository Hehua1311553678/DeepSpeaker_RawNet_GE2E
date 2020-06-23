import os
import wave
import shutil

from django.http import HttpResponse
from django.shortcuts import render

import rawnet_vctk

audio_dir_root = 'audios'

def write_wav_file(wav):
    os.makedirs(audio_dir_root, exist_ok=True)
    audio_file = audio_dir_root + '/' + wav.name

    audio = wave.open(audio_file, 'wb')
    audio.setnchannels(1)
    audio.setsampwidth(2)
    audio.setframerate(16000)
    audio.setnframes(100)
    audio.writeframes(wav.read())
    audio.close()

def delet_wav():
    if os.path.exists(audio_dir_root):
        shutil.rmtree(audio_dir_root)

def speaker_verification(request):
    ctx = {}
    if request.POST:
        if 'upload' in request.POST:
            wav1 = request.FILES.get('wav_file1', None)
            if wav1 == None:
                return HttpResponse("未上传文件！")
            wav2 = request.FILES.get('wav_file2', None)
            if wav2 == None:
                return HttpResponse("未上传文件！")
            write_wav_file(wav1)
            write_wav_file(wav2)
            ctx['success'] = '文件上传成功！可以进行测试。'
        elif 'reset' in request.POST:
            delet_wav()
        elif 'test' in request.POST:
            score = test()
            ctx['rlt'] = "得分:"+str(score)
    else:
        delet_wav()
    return render(request, 'sv.html', ctx)

def test():
    audio_files = []
    for root, dirs, files in os.walk(audio_dir_root):
        for file in files:
            audio_files.append(os.path.join(root, file))
    if len(audio_files) == 2:
        score = rawnet_vctk.sv_demo(audio_files)
        delet_wav()
        return score
    else:
        delet_wav()
        return HttpResponse("请重新上传2条音频后再点击测试！！")