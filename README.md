# MIML_project

Subtask1 运行方法:

python task1plot.py

该脚本对不同超参数画了准确率随迭代次数的图像. 图片见 ./Q1 目录 

Subtask2 运行方法:

python task2plot.py

该脚本对不同模型以及超参数画了准确率随迭代次数的图像. 图片见 ./Q2 目录

Subtask3 运行方法：

python task3plot.py

注意，该脚本会多进程调用目录中 task3_ 开头的脚本进行试验，分别对应不同的方法（如超参数，优化器等）。最后作图，作图所用的数据，子图，以及合成的大图见 ./Q3_result 目录，脚本的日志见 ./log/task_3 目录

Subtask4 运行方法：

python task4plot.py

注意，该脚本会多进程调用目录中 task4_ 开头的脚本进行试验，最后作图，图片见 ./Q4 目录，脚本的日志见 ./log/task_4 目录

Subtask5 运行方法:

q5_l2_norm.py

该脚本画了准确率以及模型参数的 l_2 范数随迭代次数的图像, q5_开头的脚本作的图均存在 ./Q5 目录

q5_jq.py

该脚本画了模型子网络的稀疏度与准确率/损失函数随迭代次数的图像

q_5_accelerate_grokking.py

该脚本画了添加扰动后准确率与模型参数的 l_2 范数随迭代次数的图像


更新日志

12.28 19：30 cfq 

1. 更新了 cfq_task3plot.py 用来画task3的图，cfq_demo_task1_Transformer_tuning.py cfq_demo_task3_RMSProp_tuning.py 分别是调参用到，可忽略。

2. dependencies.py 增加了 pandas

3. trainer.py 增加了 Trainer 类的 max_iter_step 属性，用来限制最大迭代步数，当迭代步数达到后强制结束训练，默认值为-1，表示无穷大。

4. 基于 github 仓库，更新了微信群中的  task1plot.py, task2plot.py, trainer.py 直接复制进来并覆盖。

5. trainer.py 增加了 Trainer 类的 stop_acc 属性， 默认值 99，表示会在测试集 99% 正确率停止，现在可以指定为101，不做强制停止，或者其他正确率。

12.29 23：00 cfq

1. task3 已经跑完，补充的图和中间结果存在 ./Q3_cfq_result 中，最终的图见 ./Q3_cfq_result/fig/q3_final_fig.png 。

2. 新增了 ./log 目录用来存储训练的 terminal 输出。

3. 运行方法： 运行 ./cfq_task3plot.py 即可，注意 flag_trained = True 是用现有输出来画图，flag_trained = False 会导致重新训练，覆盖上述的 result, log 中的文件。

4. 其余 ./cfq_demo_task3_xxxxxx.py 的脚本是在 ./cfq_task3plot.py 中调用的，比较了九种模型，每个命名对应一个子模型。

5. ./Q3 中之前画的图应该之后不用了，为了浏览清晰已删除。

1.1 20:00 ljq

1. model.py中的MLP模型增加了masked_forward函数，用于研究sub-network

2. trainer.py中增加了SubnetworkEvaluator类，用于寻找稀疏子网络

3. 运行q5_jq.py，可以获得在2层MLP，p=31时模型的稀疏性随训练变化结果，存储在Q5文件夹中

1.2 22:00 wrl

1. task5 计算了模型参数的l_2范数, 并在 Transformer 的 embedding 层添加扰动加速了 grokking, 图像存在Q5的accelerate起名的两个文件夹中

2. model.py 中 Transformer 的 forward 方法新增了扰动 

3. 新上传了专做 task5 的 trainer_for_task5.py

4. 上传了 task5 的运行程序 q5_l2norm_wrl.py 与 q5_accelerate_grokking_wrl.py 