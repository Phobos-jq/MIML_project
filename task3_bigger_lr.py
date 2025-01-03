from dependencies import *
from trainer import Trainer

train_data_proportion_list = [item * 0.05 for item in range(6,20)][::-1]
optimization_budget_step = 15000 # 最多使用的步数
seed_list = [10,11,12]
best_validation_acc_list = [[] for _ in range(len(seed_list))]
optimizer_type = "AdamW"
name = "bigger_lr"
for train_data_proportion in train_data_proportion_list:
    for idx, seed in enumerate(seed_list):
        logger.info(f"Training model with alpha = {train_data_proportion:.2f}, seed = {seed}")
        transformer_trainer = Trainer(
            Q4="Task3",
            p=97,
            train_data_proportion=train_data_proportion,
            random_seed=seed,
            batch_size=1024,
            verbose=1,
            num_epochs=1000,
            num_layers=2,
            seq_len=2,  # 以我的理解这是个固定值
            lr=6e-3,
            weight_decay=0.0,
            model_type="Transformer",
            lr_gamma=0.99,
            lr_step=1,
            max_iter_step=optimization_budget_step,
            optimizer_type=optimizer_type,
            dropout=0.0,
            stop_acc=100,
        )
        transformer_trainer.fit()
        transformer_trainer.plt_train_test_acc()
        stepwise_test_acc = transformer_trainer.stepwise_test_acc
        stepwise_test_acc = stepwise_test_acc[:optimization_budget_step]
        best_validation_acc = max(stepwise_test_acc)
        best_validation_acc_list[idx].append(best_validation_acc)
best_validation_acc_list = [item[::-1] for item in best_validation_acc_list]
train_data_proportion_list = train_data_proportion_list[::-1]
os.makedirs("./Q3_cfq_result/fig",exist_ok=True)
se = pd.DataFrame(np.array(best_validation_acc_list),index=seed_list,columns=train_data_proportion_list)
se.to_parquet(f"./Q3_cfq_result/df_best_val_acc_{name}.pq")

best_validation_acc_mean_list = [sum(item) / len(item) for item in zip(*best_validation_acc_list)]
plt.figure()
plt.plot(train_data_proportion_list,best_validation_acc_mean_list)
for i in range(len(seed_list)):
    plt.scatter(train_data_proportion_list,best_validation_acc_list[i],s=5,alpha=0.4)
plt.title(f"{name}")
plt.savefig(f"./Q3_cfq_result/fig/q3_{name}.png")
plt.show()