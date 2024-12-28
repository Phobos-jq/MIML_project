from trainer import Trainer  # 假设 Trainer 类在 trainer 模块中

def plot_task1_variation(p, alphas, num_epochs, batch_size, lr_values, lr_gamma_values):
    for alpha in alphas:
        for lr in lr_values:
            for lr_gamma in lr_gamma_values:
                print(f"Running Task 1 with alpha={alpha}, lr={lr}, lr_gamma={lr_gamma}")

                # 初始化 Trainer
                trainer = Trainer(
                    Q4="Task1",  # 表示 Task 1
                    p=p,
                    train_data_proportion=alpha,  # 训练数据比例
                    random_seed=42,
                    batch_size=batch_size,
                    verbose=1,
                    num_epochs=num_epochs,
                    num_layers=2,  # Transformer 默认 2 层
                    seq_len=2,  # 序列长度
                    lr=lr,  # 学习率
                    lr_gamma=lr_gamma,  # 学习率衰减因子
                    model_type="Transformer",  # 固定使用 Transformer
                    optimizer_type="AdamW",  # 固定使用 AdamW 优化器
                )

                # 训练模型
                trainer.fit()

                # 绘图并保存
                trainer.plt_train_test_acc()

                print(f"Task 1 plot saved for alpha={alpha}, lr={lr}, lr_gamma={lr_gamma}.")

# 设置 Task 1 的实验参数
p = 97  # 模数
alphas = [0.3, 0.5, 0.7]  # 训练数据比例列表
num_epochs = 200  # 训练轮数
batch_size = 512  # 每批大小
lr_values = [3e-3, 1e-3]  # 学习率列表
lr_gamma_values = [0.99, 0.95]  # 学习率衰减因子列表

# 执行绘图
plot_task1_variation(p, alphas, num_epochs, batch_size, lr_values, lr_gamma_values)
