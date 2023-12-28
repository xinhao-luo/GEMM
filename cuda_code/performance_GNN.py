from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement

import os
import optparse
import numpy as np
import matplotlib.pyplot as plt
import csv


def draw_line_chart(methods, dims, data, figure_name, y_label, title):
    fig = plt.figure(figsize=(32, 24), dpi=100)

    dims_str = list(map(str, dims))
    # print(dims_str)
    # print(methods)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    linestyles = ['-', '--', '-.', ':']
    for i in range(len(methods)):
        plt.plot(dims_str, data[i], color=colors[i % len(colors)],
                 linestyle=linestyles[(i // len(colors)) % len(linestyles)], marker='o', markersize=6)

    # plt.xticks(dims)
    plt.ylim(bottom=0)
    plt.yticks(range(0, round(np.max(np.max(data, axis=0)) + 0.5) + 1, 1))
    plt.tick_params(labelsize=25)

    # plt.hlines(y=100, xmin=dims_str[0], xmax=dims_str[-1], colors='r', linestyles='-.')
    plt.grid(True, linestyle='-.')

    plt.xlabel('dataset', fontdict={'size': '30'})
    plt.ylabel(y_label, fontdict={'size': '30'})
    plt.title(title, fontdict={'size': '30'})
    plt.legend(methods, loc='best', prop={'size': '30'})

    plt.savefig(figure_name, dpi=fig.dpi)
    # plt.show()

def draw_line_chart_time(methods, dims, data, figure_name, y_label, title):
    fig = plt.figure(figsize=(32, 24), dpi=100)

    dims_str = list(map(str, dims))
    # print(dims_str)
    # print(methods)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    linestyles = ['-', '--', '-.', ':']
    for i in range(len(methods)):
        plt.plot(dims_str, data[i], color=colors[i % len(colors)],
                 linestyle=linestyles[(i // len(colors)) % len(linestyles)], marker='o', markersize=6)

    # plt.xticks(dims)
    plt.ylim(bottom=0)
    length = []
    for i in range(51):
        length.append(0.08 * i)

    plt.yticks(length)
    plt.tick_params(labelsize=25)

    # plt.hlines(y=100, xmin=dims_str[0], xmax=dims_str[-1], colors='r', linestyles='-.')
    plt.grid(True, linestyle='-.')

    plt.xlabel('dataset', fontdict={'size': '30'})
    plt.ylabel(y_label, fontdict={'size': '30'})
    plt.title(title, fontdict={'size': '30'})
    plt.legend(methods, loc='best', prop={'size': '30'})

    plt.savefig(figure_name, dpi=fig.dpi)
    # plt.show()

def read_data(data_path, log_files):
    data_throughput = []
    data_time = []
    for log_file in log_files:
        data = []
        with open(data_path + log_file) as fp:
            reader = csv.reader(fp)
            next(reader)  # 跳过标题行
            for row in reader:
                data.append(row)
        # print(data)

        data_throughput.append([float(row[4]) for row in data])
        data_time.append([float(row[5].split('ms')[0]) for row in data])
    # print(data_throughput)
    return data_throughput, data_time


def get_dims(log_files):
    dims = ['cora', 'citeseer', 'pubmed', 'ppi', 'amazon0505','artist','com-ama','soc', 'amazon0601','DD','TWITTER']

    return dims


def get_methods(log_file):
    methods = []

    for method in log_file:
        methods.append(method.split('.csv')[0])
    # print(methods)
    return methods


def analyze_data(data_path):
    log_files = []
    for file_name in os.listdir(data_path):
        if '.csv' not in file_name:
            continue

        log_files.append(file_name)
    # print(log_files)
    methods = get_methods(log_files)
    dims = get_dims(log_files)
    data_throughput, data_time = read_data(data_path, log_files)
    draw_line_chart(methods, dims, data_throughput, data_path +
                    'gemm_throughput_GNN.png', 'Throughput / TFLOPS', 'GEMM Throughput')
    draw_line_chart_time(methods, dims, data_time, data_path +
                    'gemm_time_GNN.png', 'Time / ms', 'GEMM Time')


usage = "python3 performance.py -p/--path exp_new/GNN/"
parser = optparse.OptionParser(usage)
parser.add_option('-p', '--path', dest='path',
                    type='string', help='data path', default='exp_new/GNN/')

options, args = parser.parse_args()
path = options.path
# print(path)
analyze_data(path)
