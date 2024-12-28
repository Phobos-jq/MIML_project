from dependencies import *
from trainer import Trainer

transformer_trainer = Trainer(
    Q4="Task3",
    p=97,
    train_data_proportion=0.5,
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
)
transformer_trainer.fit()
transformer_trainer.plt_train_test_acc()
