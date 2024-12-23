from dependencies import *
from trainer import Trainer

transformer_trainer = Trainer(
    p=97,
    train_data_proportion=0.5,
    random_seed=10,
    batch_size=int(99*99*0.5),
    verbose=1,
    num_epochs=1000,
    num_layers=2,
    seq_len=2, # 以我的理解这是个固定值
    lr=3e-3,
    weight_decay=1.0,
    model_type="Transformer",
    MLP_layer_dims=[512, 1024, 1024, 512, 128],
    lr_gamma=0.99,
)

transformer_trainer.fit()
transformer_trainer.plt_train_test_acc()