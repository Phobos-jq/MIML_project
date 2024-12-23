from dependencies import *
from trainer import Trainer
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import itertools




# 定义训练函数，接受 p 和 k 参数
def run_training(p, k, lr, wd, dropout):
    # 为每个子进程设置独立的日志文件
    log_file_name = f"log_p{p}_k{k}_lr{lr}_wd_{wd}_dropout{dropout}.log"
    handler_id = logger.add(log_file_name, format="{time} {level} {message}", level="INFO", rotation="10 MB", mode="w")

    try:
        logger.success(f"Running training for p={p}, k={k}")
        model_type = "Transformer"
        optimizer_type = "AdamW"
        lr_gamma = 0.995
        lr_step = 50
        momentum = 0.9
        nesterov = False
        dropout = 0.2
        batch_size = 2**14
        num_epochs = 100000

        train_data_proportions = [0.3 + i * 0.2 for i in range(4)]
        accs = []  # 记录当前组合下的精度
        for fraction in train_data_proportions:
            eval_step = min(16, max(1, int(((p**k) * fraction / batch_size) / 16)))
            data_size = num_epochs * fraction
            transformer_trainer = Trainer(
                p=p,
                k=k,
                Q4=True,
                train_data_proportion=fraction,
                random_seed=10,
                batch_size=batch_size,
                verbose=2,
                eval_step=eval_step,
                num_epochs=num_epochs,
                lr=lr,
                weight_decay=wd,
                model_type=model_type,
                optimizer_type=optimizer_type,
                lr_gamma=lr_gamma,
                lr_step=lr_step,
                momentum=momentum,
                nesterov=nesterov,
                dropout=dropout,
            )
            transformer_trainer.fit()
            final_acc = transformer_trainer.plt_train_test_acc()
            accs.append(final_acc)
            logger.success(f"Finished training for p={p}, k={k}, alpha={fraction}")

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
            f'Q4/p_{p}__k_{k}__model_{model_type}__optim_{optimizer_type}__lr_{lr}__wd_{wd}/'
            f'dropout_{dropout}__q4_momentum_{momentum}__nesterov_{nesterov}__lrGamma_{lr_gamma}__lrStep_{lr_step}__bs_{batch_size}__ds_{data_size}.png'
        )
        plt.close()

    except Exception as e:
        logger.error(f"Error occurred for p={p}, k={k}: {e}")
    
    finally:
        # 移除当前进程的日志 handler
        logger.remove(handler_id)

# 参数列表
p_list = [31]
k_list = [2, 3, 4]
learning_rates = [1e-3, 3e-3, 3e-4]
weight_decays = [1.0]
dropouts = [0.2]
max_workers = 9


experiment_params = [
    (p, k, lr, wd, dropout)
    for p in p_list
    for k in k_list
    for lr in learning_rates
    for wd in weight_decays
    for dropout in dropouts
]

# 并行运行任务
if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务给进程池
        futures = [executor.submit(run_training, *params) for params in experiment_params]

        # 等待所有任务完成
        for future in futures:
            try:
                future.result()  # 捕获可能的异常
            except Exception as e:
                logger.error(f"Exception in parallel training: {e}")