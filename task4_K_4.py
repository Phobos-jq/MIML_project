from dependencies import *
from trainer import Trainer
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import itertools

# K=4
transformer_trainer = Trainer(
    p=17,
    k=4,
    Q4="Task4",
    train_data_proportion=0.7,
    random_seed=10,
    batch_size=int(17**4*0.7),
    verbose=1,
    eval_step=1,
    num_epochs=10000,
    lr=1e-3,
    weight_decay=0.0,
    model_type="Transformer",
    optimizer_type="AdamW",
    lr_gamma=0.995,
    lr_step=50,
    momentum=0.8,
    nesterov=False,
    dropout=0.0,
    stop_acc=96,
)

transformer_trainer.fit()
final_acc = transformer_trainer.plt_train_test_acc()
