
def read_scp_lines(scp_dir, txt_file):
    with open(scp_dir, encoding='utf-8', mode='r') as scp_f:
        scp_lines = scp_f.readlines()
    scp_f.close()
    results = []
    for line in scp_lines:
        k, f, p = line.strip().split(' ')
        results.append(k+'\n')
    print("len:{}".format(len(results)))
    with open(txt_file, encoding='utf-8', mode='w') as txt_f:
        txt_f.writelines(results)
    txt_f.close()
    print("finished read_scp_lines!")

def couple(txt_file):
    with open(txt_file, encoding='utf-8', mode='r') as txt_f:
        lines = txt_f.readlines();
    txt_f.close()
    num = len(lines)
    print("len:{}".format(num))
    i = 0
    results = []
    for item in lines:
        i += 10
        i %= num
        results.append(item.strip()+" "+lines[i].strip()+'\n')
    print("len:{}".format(len(results)))
    with open(txt_file, encoding='utf-8', mode='w') as txt_f:
        txt_f.writelines(results)
    txt_f.close()
    print("finished couple!")

def compare(txt_file):
    with open(txt_file, encoding='utf-8', mode='r') as txt_f:
        lines = txt_f.readlines();
    txt_f.close()
    results = []
    num = 0
    for item in lines:
        a, b = item.strip().split(' ')
        k = 0
        if a.split('/')[1] == b.split('/')[1]:
            k = 1
        num += k
        results.append(str(k)+' '+a+' '+b+'\n')
    print("len:{}, 其中匹配个数:{},不匹配个数：{}".format(len(results), num, len(results)-num))
    with open(txt_file, encoding='utf-8', mode='w') as txt_f:
        txt_f.writelines(results)
    txt_f.close()
    print("finished compare!")

def liangji():
    while 1:
        print("踹死你！！！！！\n")

UttPerSpk = '_spk5_UttPerSpk5'

test_file = "trials"+UttPerSpk+".txt"
dev_file = "val_trials"+UttPerSpk+".txt"
test_scp_dir = "scp\\test\\test_wav_pe"+UttPerSpk+".scp"
dev_scp_dir = "scp\\dev\dev_wav_pe"+UttPerSpk+".scp"


# scp_dir = test_scp_dir
# txt_file = test_file
scp_dir = dev_scp_dir
txt_file = dev_file

read_scp_lines(scp_dir, txt_file)
couple(txt_file)
compare(txt_file)