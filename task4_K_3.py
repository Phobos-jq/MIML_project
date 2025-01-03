from dependencies import *
from trainer import Trainer
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import itertools

# K=3 by jq
transformer_trainer = Trainer(
    p=31,
    k=3,
    Q4="Task4",
    train_data_proportion=0.3,
    random_seed=10,
    batch_size=int(2**14),
    verbose=1,
    eval_step=min(16, max(1, int(((31**3) * 0.3 / 2**14) / 16))),
    num_epochs=100000,
    lr=3e-4,
    weight_decay=1.0,
    model_type="Transformer",
    optimizer_type="AdamW",
    lr_gamma=0.995,
    lr_step=50,
    momentum=0.8,
    nesterov=False,
    dropout=0.2,
    stop_acc=99,
)

transformer_trainer.fit()
final_acc = transformer_trainer.plt_train_test_acc()