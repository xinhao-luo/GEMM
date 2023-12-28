import os
import argparse
import subprocess
import pandas as pd

dataset = [
        ('cora',2708 	        	, 1433	    , 7   ),  
        ('citeseer',3327	        , 3703	    , 6   ),  
        ('pubmed',19717	        	, 500	    , 3   ),      
        ('ppi',56944	            , 50	    , 121 ),   

        ( 'amazon0505',410236               , 96	  , 22),
        ( 'artist',50515                   , 100    , 12),
        ( 'com-amazon',334863              , 96	  , 22),
        ( 'soc-BlogCatalog',88784	         , 128    , 39), 
        ( 'amazon0601',403394  	         , 96	  , 22), 

        ('PROTEINS_full',43466            , 29       , 2) ,   
        ('OVCAR-8H',1889542                 , 66       , 2) , 
        ('Yeast',1710902                   , 74       , 2) ,
        ('DD',334925                     , 89       , 2) ,
        ('TWITTER-Real-Graph-Partial',580768, 1323     , 2) ,   
        ('SW-620H',1888584                  , 66       , 2) 
]


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

    fout = open(log.strip(".log") + ".csv", 'a+')
    fout.write("dataset,M,N,K,GFLOPs,Time (ms)\n")
    for data, ops, time in zip(dataset, ops_li, time_li):
        data_name, num_vertexes, in_feat, out_feat = data
        fout.write("{},{},{},{},{},{}".format(data_name, num_vertexes, 16, in_feat, ops, time))
    fout.close()

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="0", help="device id")

# parser.add_argument("--gpu_type", default="V100", help="gpu type", type=str)
parser.add_argument("--test_file", default="cublas", help="test file", type=str)

args = parser.parse_args()
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(CURRENT_DIR, 'exp')
if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)
log = os.path.join(LOG_DIR,f'{args.test_file}_GNN.log')
for data, num_vertexes, in_feat, out_feat in dataset:
    command = f"nvcc -arch=sm_80 src/{args.test_file}.cu src/main.cu -o test && ./test {num_vertexes} {16} {in_feat}"
    with open(log,'a') as fp:
        ret = subprocess.call(command, shell=True, stdout=fp)
log2csv(log)