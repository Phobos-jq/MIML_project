# MIML_project

## 主要代码内容
* data.py: 生成训练与测试数据集
* model.py: 定义模型,包括Transformer，MLP，LSTM
* trainer.py: 定义训练器，包括训练，验证，测试，子网络搜索等函数
* requirement.txt: 项目所需的库
* 目录Q1至Q5：分别存放了不同子任务的图片，log目录下存放了不同子任务的日志
* 剩余的脚本实例化了模型并用多种优化器对模型进行训练，用于完成不同的子任务。具体见运行方式


## 运行方式
### Subtask1 运行方法:
    python task1plot.py
该脚本对不同超参数画了准确率随迭代次数的图像. 图片见 ./Q1 目录 

### Subtask2 运行方法:
    python task2plot.py
该脚本对不同模型以及超参数画了准确率随迭代次数的图像. 图片见 ./Q2 目录

### Subtask3 运行方法：
    python task3plot.py
注意，该脚本会多进程调用目录中 task3_ 开头的脚本进行试验，分别对应不同的方法（如超参数，优化器等）。最后作图，作图所用的数据，子图，以及合成的大图见 ./Q3_result 目录，脚本的日志见 ./log/task_3 目录

### Subtask4 运行方法：
    python task4plot.py
注意，该脚本会多进程调用目录中 task4_ 开头的脚本进行试验，最后作图，图片见 ./Q4 目录，脚本的日志见 ./log/task_4 目录

### Subtask5 运行方法:
    python q5_l2_norm.py
该脚本画了准确率以及模型参数的 l_2 范数随迭代次数的图像, q5_开头的脚本作的图均存在 ./Q5 目录
    
    python q5_jq.py
该脚本画了模型子网络的稀疏度与准确率/损失函数随迭代次数的图像
    
    python q_5_accelerate_grokking.py
该脚本画了添加扰动后准确率与模型参数的 l_2 范数随迭代次数的图像