from dependencies import *
from trainer import Trainer

train_data_proportion_list = [item * 0.05 for item in range(6,20)]
optimization_budget_step = 100000 # 最多使用的步数
best_validation_acc_list = []
for train_data_proportion in train_data_proportion_list:
    transformer_trainer = Trainer(
        Q4="Task3",
        p=97,
        train_data_proportion=train_data_proportion,
        random_seed=42,
        batch_size=512,
        verbose=1,
        num_epochs=200,
        num_layers=2,
        seq_len=2,  # 以我的理解这是个固定值
        lr=3e-3,
        weight_decay=0.0,
        model_type="Transformer",
        lr_gamma=0.99,
        lr_step=1,
        optimizer_type="RMSprop",
        momentum=0.0,
        dropout=0.0,
    )
    transformer_trainer.fit()

    stepwise_test_acc = transformer_trainer.stepwise_test_acc
    stepwise_test_acc = stepwise_test_acc[:optimization_budget_step]
    best_validation_acc = max(stepwise_test_acc)
    best_validation_acc_list.append(best_validation_acc)

plt.figure()
plt.plot()
