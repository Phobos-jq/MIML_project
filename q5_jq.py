from dependencies import *
from trainer import Trainer

transformer_trainer = Trainer(
    p=31,
    train_data_proportion=0.7,
    random_seed=42,
    batch_size=512,
    verbose=1,
    num_epochs=2000,
    num_layers=2,
    seq_len=2, # 以我的理解这是个固定值
    lr=3e-3,
    weight_decay=1.0,
    model_type="MLP",
    MLP_layer_dims=[64,64],
    lr_gamma=0.99,
    Q4="test",
    dropout=0,
    subnetwork=True,
    threshold=0.98,
)

transformer_trainer.fit()
transformer_trainer.plt_train_test_acc()
transformer_trainer.plt_train_test_acc_sparsity()
transformer_trainer.plt_train_test_loss_sparsity()
