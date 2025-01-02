from trainer_for_task5 import Trainer
import logging

alpha=0.3
batch_size=512
num_epochs=1000
lr=3e-3
lr_gamma=0.99
weight_decay=1

trainer = Trainer(
    use_adaptive_noise=True,
    noise_lambda1=0.1,  # 论文中模运算任务使用的值
    noise_lambda2=0.08,
    # ... 其他参数 ...
    Q4="Task5",  # 表示 Task 1
    p=97,
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
    weight_decay=weight_decay
)
trainer.fit()

final_test_acc = trainer.plt_train_test_acc()

logging.info(f"Training completed. Final test accuracy: {final_test_acc:.4f}")
