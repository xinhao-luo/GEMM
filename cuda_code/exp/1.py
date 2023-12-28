import matplotlib.pyplot as plt
import csv

# 读取CSV文件
data = []
with open('gemm_v0.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # 跳过标题行
    for row in reader:
        data.append(row)

# 提取数据列
# for row in data:
#     print(row[4].split('ms')[0])
sizes = [int(row[0]) for row in data]
times = [float(row[4].split('ms')[0]) for row in data]

# 绘制折线图
plt.plot(sizes, times, marker='o')

# 添加标题和标签
plt.title('Performance vs. Size')
plt.xlabel('Matrix Dimension / M = N = K')
plt.ylabel('Time (ms)')

plt.xticks(range(min(sizes), max(sizes)+20, 128))

# 显示图形
# 保存图形为文件
plt.savefig('line_plot.png')