import os
from torch import nn, optim
from torch.utils.data import DataLoader
from catalyst import dl, utils
from catalyst.contrib.datasets import MNIST

# model configuration
model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)
loaders = {
    "train": DataLoader(MNIST(os.getcwd(), train=True), batch_size=32),
    "valid": DataLoader(MNIST(os.getcwd(), train=False), batch_size=32),
}
runner = dl.SupervisedRunner(
    input_key="features", output_key="logits", target_key="targets", loss_key="loss"
)

# model training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loaders=loaders,
    num_epochs=1,
    callbacks=[
        dl.AccuracyCallback(input_key="logits", target_key="targets", topk=(1,3,5)),
        dl.PrecisionRecallF1SupportCallback(input_key="logits", target_key="targets"),
    ],
    logdir="./logs",
    valid_loader="valid",
    valid_metric="loss",
    minimize_valid_metric=True,
    verbose=True,
)
