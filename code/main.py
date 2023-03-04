# -*- coding:utf-8 -*-

import pytorch_lightning as pl
from datamodule import TSDataModule
from model import LitLSTM, LSTMModel
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn


if __name__ == "__main__":
    path = "data/input_data.csv"
    num_time_steps = 6

    batch_size = 32
    learning_rate = 1e-2

    num_features = 9
    hidden_size = 16
    num_lstm_layers = 1
    bidirectional = False

    ts_data_module = TSDataModule(
        path, num_time_steps=num_time_steps, batch_size=batch_size
    )

    lstm_net = LSTMModel(
        input_size=num_features,
        hidden_size=hidden_size,
        num_layers=num_lstm_layers,
        bidirectional=bidirectional,
    )
    loss_func = nn.MSELoss()
    pl_net = LitLSTM(lstm_net, loss_func)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        filename="lstm-{epoch:04d}-{val_loss:.8f}",
    )
    trainer = pl.Trainer(callbacks=[checkpoint_callback])
    trainer.fit(model=pl_net, datamodule=ts_data_module)
