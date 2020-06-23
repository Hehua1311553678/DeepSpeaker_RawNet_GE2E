import logging
import os
import re
from glob import glob
import matplotlib.pyplot as plt
# import constants as c
import logging
import os
import re
from glob import glob
import matplotlib.pyplot as plt
# import constants as c
import constants_vctk as c#===================================================2020/04/17 20:59
# from utils import plot_loss, plot_acc

def plot_loss(file=c.CHECKPOINT_FOLDER+'/losses.txt'):
    step = []
    loss = []
    dict_step_loss = {}
    mov_loss = []
    ml = 0
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            dict_step_loss[int(line.split(',')[0])] = line
    for key in dict_step_loss.keys():
        line = dict_step_loss[key]
        step.append(int(line.split(",")[0]))
        loss.append(float(line.split(",")[1]))
        if ml == 0:
            ml = float(line.split(",")[1])
        else:
            ml = 0.01 * float(line.split(",")[1]) + 0.99 * mov_loss[-1]
        mov_loss.append(ml)
    p1, = plt.plot(step, loss)
    p2, = plt.plot(step, mov_loss)
    plt.legend(handles=[p1, p2], labels = ['loss', 'moving_average_loss'], loc = 'best')
    plt.xlabel("Iters")
    plt.ylabel("Losses")
    plt.show()

def plot_acc(file=c.CHECKPOINT_FOLDER+'/acc_eer.txt'):
    step = []
    eer = []
    fm = []
    acc = []
    dict_step_line = {}
    mov_eer=[]
    mv = 0
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            dict_step_line[int(line.split(",")[0])] = line
    for key in dict_step_line.keys():
        line = dict_step_line[key]
        step.append(int(line.split(",")[0]))
        eer.append(float(line.split(",")[1]))
        fm.append(float(line.split(",")[2]))
        acc.append(float(line.split(",")[3]))
        if mv == 0:
           mv = float(line.split(",")[1])
        else:
           mv = 0.1*float(line.split(",")[1]) + 0.9*mov_eer[-1]
        mov_eer.append(mv)

    # ===========================================================================================2020/04/17 00:44
    # p1, = plt.plot(step, fm, color='black',label='F-measure')
    # p2, = plt.plot(step, eer, color='blue', label='EER')
    # p3, = plt.plot(step, acc, color='red', label='Accuracy')
    # p4, = plt.plot(step, mov_eer, color='red', label='Moving_Average_EER')
    p1, = plt.plot(step, fm, 'g+--',label='F-measure')
    p2, = plt.plot(step, eer, 'b^--', label='EER')
    p3, = plt.plot(step, acc, 'ro--', label='Accuracy')
    p4, = plt.plot(step, mov_eer, color='black', label='Moving_Average_EER')
    # ===========================================================================================
    plt.xlabel("Steps")
    # plt.ylabel("I dont know")
    plt.legend(handles=[p1,p2,p3,p4],labels=['F-measure','EER','Accuracy','moving_eer'],loc='best')
    plt.show()

CHECKPOINT_FOLDER = 'vctk_UttPerSpk5/checkpoints'
loss_file=CHECKPOINT_FOLDER+'/losses.txt' # loss file path
plot_loss(loss_file)
# acc_file=CHECKPOINT_FOLDER+'/train_acc_eer.txt' # loss file path
# plot_acc(acc_file)
# loss_file=c.CHECKPOINT_FOLDER+'/losses.txt' # loss file path
# plot_loss(loss_file)
# acc_file=c.CHECKPOINT_FOLDER+'/train_acc_eer.txt' # loss file path
# plot_acc(acc_file)