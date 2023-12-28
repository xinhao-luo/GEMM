import os
import argparse
import subprocess
import pandas as pd

size_all = [128, 256, 512, 768, 1024, 2048, 3072, 4096, 5120, 6144, 7168]

def log2csv(log):
    ops_li = []
    time_li = []
    fp = open(log, "r")
    for line in fp:
        if " Gops" in line:
            ops = line[0: line.index(' ')]
            # print(data)
            ops_li.append(ops)
        pattern = 'Time: '
        if pattern in line:
            time = line.split(pattern)[1].rstrip("ms")
            # print(time)
            time_li.append(time)
    fp.close()

    fout = open(log.strip(".log") + ".csv", 'w')
    fout.write("M,N,K,TFLOPs,Time (ms)\n")
    for size, ops, time in zip(size_all, ops_li, time_li):
        fout.write("{},{},{},{},{}".format(size, size, size, ops, time))
    fout.close()

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="0", help="device id")

# parser.add_argument("--gpu_type", default="A100", help="gpu type", type=str)
parser.add_argument("--test_file", default="cublas", help="test file", type=str)

args = parser.parse_args()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(CURRENT_DIR, 'exp_new')
if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)
log = os.path.join(LOG_DIR,f'{args.test_file}.log')
for size in size_all:
    command = f"./{args.test_file} {size} {size} {size}"
    with open(log,'a') as fp:
        ret = subprocess.call(command, shell=True, stdout=fp)
log2csv(log)

