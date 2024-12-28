from trainer import Trainer

# 调节的参数
train_data_proportion = 0.6  # 训练数据比例
lr = 1e-3  # 学习率
lr_gamma = 0.99  # 学习率衰减因子
weight_decay = 1  # 权重衰减
num_epochs = 1000  # 训练轮数

# 创建并训练 MLP 模型
print("Training MLP Model...")
mlp_trainer = Trainer(
    model_type="MLP",  # 指定使用 MLP 模型
    train_data_proportion=train_data_proportion,
    lr=lr,
    lr_gamma=lr_gamma,
    weight_decay=weight_decay,
    num_epochs=num_epochs,
    verbose=1,
    Q4 = "Task2"
)
mlp_trainer.fit()
mlp_trainer.plt_train_test_acc()  # 绘制准确性曲线

# 创建并训练 LSTM 模型
print("Training LSTM Model...")
lstm_trainer = Trainer(
    model_type="LSTM",  # 指定使用 LSTM 模型
    train_data_proportion=train_data_proportion,
    lr=lr,
    lr_gamma=lr_gamma,
    weight_decay=weight_decay,
    num_epochs=num_epochs,
    verbose=1,
    Q4 = "Task2"
)
lstm_trainer.fit()
lstm_trainer.plt_train_test_acc()  # 绘制准确性曲线
