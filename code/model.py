# -*- coding:utf-8 -*-

import torch
from torch import nn
import pytorch_lightning as pl


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs
    ):
        super(LSTMModel, self).__init__(**kwargs)
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
            bias=bias,
        )
        self.output_linear = nn.Linear(
            2 * hidden_size if bidirectional else hidden_size, 1
        )

    def forward(self, x):
        """
        x shape is (B, T, C)

        output shape is (B,)

        """

        return self.output_linear(self.rnn(x)[0][:, -1, :]).squeeze(axis=1)


class LitLSTM(pl.LightningModule):
    def __init__(self, lstm_net, loss_func, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lstm_net = lstm_net
        self.loss_func = loss_func
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.lstm_net(x)

    def forward_with_loss(self, x, y):
        pred = self.forward(x)
        return self.loss_func(pred, y)

    def training_step(self, batch, batch_idx):
        loss_value = self.forward_with_loss(*batch)
        self.log("training_loss", loss_value)
        return loss_value

    def validation_step(self, batch, batch_idx):
        loss_value = self.forward_with_loss(*batch)
        self.log("val_loss", loss_value)
        return loss_value

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    lstm_net = LSTMModel(input_size=3, hidden_size=64, num_layers=2, bidirectional=True)
    mse_loss = nn.MSELoss()
    pl_net = LitLSTM(lstm_net, mse_loss)
