from pytorch_lightning import LightningModule
from torch import argmax
from torch.nn import BatchNorm1d, Conv1d, Linear, MaxPool1d, Module, Sequential
from torch.nn.functional import cross_entropy, relu
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.classification.f_beta import F1Score


class Lambda(Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class ResidualBlock(Module):
    """
    The residual block used by ProtCNN (https://www.biorxiv.org/content/10.1101/626507v3.full).

    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first convolution
        dilation: Dilation rate of the first convolution
    """

    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()

        self.skip = Sequential()

        self.bn1 = BatchNorm1d(in_channels)
        self.conv1 = Conv1d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            bias=False,
                            dilation=dilation,
                            padding=dilation)

        self.bn2 = BatchNorm1d(out_channels)
        self.conv2 = Conv1d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            bias=False,
                            padding=1)

    def forward(self, x):
        activation = relu(self.bn1(x))
        x1 = self.conv1(activation)
        x2 = self.conv2(relu(self.bn2(x1)))

        return x2 + self.skip(x)


class ProtCNN(LightningModule):
    def __init__(self, num_id, num_classes, params):
        super().__init__()
        self.model = Sequential(
            Conv1d(num_id, 128, kernel_size=1, padding=0, bias=False),
            ResidualBlock(128, 128, dilation=2),
            ResidualBlock(128, 128, dilation=3),
            MaxPool1d(3, stride=2, padding=1),
            Lambda(lambda x: x.flatten(start_dim=1)),
            Linear(7680, num_classes)
        )

        self.train_f1 = F1Score()
        self.valid_f1 = F1Score()
        self.test_f1 = F1Score()
        self.lr = params['learning_rate']
        self.momentum = params['momentum']
        self.weight_decay = params['weight_decay']
        self.gamma = params['gamma']
        self.milestones = list(range(5, params['epochs'] - 5, 2))

    def forward(self, x):
        return self.model(x.float())

    def training_step(self, batch, batch_idx):
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        pred = argmax(y_hat, dim=1)
        self.train_f1(pred, y)
        self.log('train_f1', self.train_f1, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        pred = argmax(y_hat, dim=1)
        acc = self.valid_f1(pred, y)
        self.log('valid_f1', self.valid_f1, on_step=False, on_epoch=True)

        return acc

    def test_step(self, batch, batch_idx):
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        pred = argmax(y_hat, dim=1)
        acc = self.test_f1(pred, y)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)

        return acc

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(),
                        lr=self.lr,
                        momentum=self.momentum,
                        weight_decay=self.weight_decay)
        lr_scheduler = MultiStepLR(optimizer,
                                   milestones=self.milestones,
                                   gamma=self.gamma)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
