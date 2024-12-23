from dependencies import *
from trainer import Trainer
import matplotlib.pyplot as plt

model_type = "Transformer"
optimizer_type = "AdamW"
lr_gamma = 0.99
lr_step = 1
momentum = 0.9
nesterov = False
batch = "mini_batch"        # "full_batch" or "mini_batch"
data_num = 400             # the whole training data size is data_num*dataset_size(97*97)

# 初始化参数
learning_rates = [1e-2, 3e-2, 3e-3]
weight_decays = [0.05, 1.0]
train_data_proportions = [0.25 + i * 0.05 for i in range(14)]

# 实验循环
for lr in learning_rates:
    if lr == 1e-2:
        for wd in weight_decays:
            accs = []  # 记录当前组合下的精度
            for fraction in train_data_proportions:
                transformer_trainer = Trainer(
                    p=97,
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
                    optimizer_type=optimizer_type,
                    lr_gamma=lr_gamma,
                    lr_step=lr_step,
                    momentum=momentum,
                    nesterov=nesterov,
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
                f"Q3/model_{model_type}__optim_{optimizer_type}__lr_{lr}__wd_{wd}/"
                f"q3_momentum_{momentum}__nesterov_{nesterov}__lrGamma_{lr_gamma}__lrStep_{lr_step}__bs_{batch}__ds_{data_num}.png.png"
            )
            plt.close()
    else:
        wd = 0
        accs = []  # 记录当前组合下的精度
        for fraction in train_data_proportions:
            transformer_trainer = Trainer(
                p=97,
                train_data_proportion=fraction,
                random_seed=10,
                batch_size=int(99 * 99 * fraction) if batch=="full_batch" else 512,
                verbose=1,
                num_epochs=int(data_num/fraction),
                num_layers=2,
                seq_len=2,  # 固定值
                lr=lr,
                weight_decay=wd,
                model_type=model_type,
                optimizer_type=optimizer_type,
                lr_gamma=lr_gamma,
                lr_step=lr_step,
                momentum=momentum,
                nesterov=nesterov,
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
            f"Q3/model_{model_type}__optim_{optimizer_type}__lr_{lr}__wd_{wd}/"
            f"q3_momentum_{momentum}__nesterov_{nesterov}__lrGamma_{lr_gamma}__lrStep_{lr_step}__bs_{batch}__ds_{data_num}.png.png"
        )
        plt.close()
