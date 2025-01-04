from trainer_for_task5 import Trainer
import logging
import sys

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def run_grokking_experiment():
    # 设置实验参数
    experiment_params = {
        'p': 97,                     # 模数
        'k': 2,                      # 加数个数
        'Q4': "Task5",              # 指定为Task5
        'train_data_proportion': 0.3, # 使用较小的训练集比例
        'random_seed': 42,
        'batch_size': 512,
        'verbose': 1,
        'num_epochs': 1000,          # 使用较长的训练时间
        'num_layers': 2,
        'lr': 3e-3,
        'weight_decay': 1,        # 使用较大的权重衰减
        'model_type': "Transformer",  # 使用Transformer模型
        'optimizer_type': "AdamW",
        'lr_gamma': 0.99,             # 保持固定学习率
    }

    # 创建trainer实例
    trainer = Trainer(**experiment_params)
    
    # 训练模型并记录结果
    trainer.fit()
    
    # 绘制结果（包含L2范数）
    final_test_acc = trainer.plt_train_test_acc()
    
    logging.info(f"Training completed. Final test accuracy: {final_test_acc:.4f}")

if __name__ == "__main__":
    run_grokking_experiment()