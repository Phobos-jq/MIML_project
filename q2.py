from dependencies import *
from trainer import Trainer
import matplotlib.pyplot as plt
import multiprocessing as mp

max_workers = 12
nesterov=False
dampening=0.0
optim = "AdamW"
lr_gamma = 0.99
lr_step = 50
momentum = 0.9
batch = "full_batch"        # "full_batch" or "mini_batch"
data_num = 20000             # the whole training data size is data_num*dataset_size(97*97)

# 初始化参数
model_types = ["MLP"]
MLP_layer_dimss = [[512, 128, 64, 64],[512, 128, 64]]
learning_rates = [3e-3, 3e-4]
weight_decays = [1.0]
dropouts = [0]
train_data_proportions = [0.3 + i * 0.2 for i in range(4)]

def run_experiment(model_type, lr, wd, dropout):
    """单个实验任务"""
    log_file_name = f"model_{model_type}__optim{optim}_lr{lr}_wd{wd}_dropout{dropout}.log"
    handler_id = logger.add(log_file_name, format="{time} {level} {message}", level="INFO", rotation="30 MB", mode="w")
    try:
        accs = []  # 记录当前组合下的精度
        for fraction in train_data_proportions:
            transformer_trainer = Trainer(
                p=97,
                Q4=False,
                train_data_proportion=fraction,
                random_seed=10,
                batch_size=int(99 * 99 * fraction) if batch=="full_batch" else 512,
                verbose=1,
                num_epochs=int(data_num/fraction), # 根据数据量调整epoch数
                num_layers=2,
                seq_len=2,  # 固定值
                lr=lr,
                weight_decay=wd,
                model_type=model_type,
                optimizer_type=optim,
                lr_gamma=lr_gamma,
                lr_step=lr_step,
                momentum=momentum,
                dropout=dropout,
            )
            transformer_trainer.fit()
            final_acc = transformer_trainer.plt_train_test_acc()
            accs.append(final_acc)
        # 绘制当前参数组合的图像
        plt.figure(figsize=(8, 6))
        plt.plot(train_data_proportions, accs, marker="o")
        plt.title(f"Learning rate={lr}, Weight decay={wd}")
        plt.xlabel("Training data fraction")
        plt.ylabel("Best validation accuracy")
        plt.ylim(0, 1.1)
        plt.grid(True)

        # 保存图像
        plt.savefig(
            f"Q3/model_{model_type}__optim_{optim}__lr_{lr}__wd_{wd}/"
            f"dropout_{dropout}__q3_momentum_{momentum}__nesterov_{nesterov}__dampening_{dampening}__lrGamma_{lr_gamma}__lrStep_{lr_step}__bs_{batch}__ds_{data_num}.png.png"
        )
        plt.close()

    except Exception as e:
        logger.error(f"Error occurred for parameters {lr=}, {wd=}, {optim=}, {dropout=}: {e}")
    
    finally:
        # 移除当前进程的日志 handler
        logger.remove(handler_id)

def main():
    # 创建参数组合
    experiment_params = [
        (model_type, lr, wd, dropout,)
        for model_type in model_types
        for lr in learning_rates
        for wd in weight_decays
        for dropout in dropouts
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务给进程池
        futures = [executor.submit(run_experiment, *params) for params in experiment_params]

        # 等待所有任务完成
        for future in futures:
            try:
                future.result()  # 捕获可能的异常
            except Exception as e:
                logger.error(f"Exception in parallel training: {e}")


if __name__ == "__main__":
    main()