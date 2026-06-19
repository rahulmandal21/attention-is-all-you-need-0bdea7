import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from typing import Callable, Iterable

class TrainingLoop:
    """
    A class used to manage the training loop of a PyTorch model.

    Attributes:
    ----------
    model : nn.Module
        The PyTorch model to be trained.
    dataloader : DataLoader
        The data loader for the training data.
    loss_fn : Callable
        The loss function to be used for training.
    device : torch.device
        The device on which the model will be trained.
    max_grad_norm : float
        The maximum gradient norm for gradient clipping.
    warmup_steps : int
        The number of warmup steps for the learning rate schedule.
    d_model : int
        The dimensionality of the model.
    """

    def __init__(self, model: nn.Module, dataloader: DataLoader, loss_fn: Callable, device: torch.device, max_grad_norm: float = 1.0, warmup_steps: int = 1000, d_model: int = 512):
        """
        Initializes the TrainingLoop class.

        Parameters:
        ----------
        model : nn.Module
            The PyTorch model to be trained.
        dataloader : DataLoader
            The data loader for the training data.
        loss_fn : Callable
            The loss function to be used for training.
        device : torch.device
            The device on which the model will be trained.
        max_grad_norm : float, optional
            The maximum gradient norm for gradient clipping (default is 1.0).
        warmup_steps : int, optional
            The number of warmup steps for the learning rate schedule (default is 1000).
        d_model : int, optional
            The dimensionality of the model (default is 512).
        """
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = LambdaLR(self.optimizer, self.lr_schedule)

    def lr_schedule(self, step_num: int) -> float:
        """
        Calculates the learning rate at a given step number.

        Parameters:
        ----------
        step_num : int
            The current step number.

        Returns:
        -------
        float
            The learning rate at the given step number.
        """
        return self.d_model ** -0.5 * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.5)

    def train_one_epoch(self) -> float:
        """
        Trains the model for one epoch.

        Returns:
        -------
        float
            The average loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        for batch in self.dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
        return total_loss / len(self.dataloader)

    def train(self, num_epochs: int) -> Iterable[float]:
        """
        Trains the model for a specified number of epochs.

        Parameters:
        ----------
        num_epochs : int
            The number of epochs to train the model for.

        Yields:
        -------
        float
            The average loss over each epoch.
        """
        for _ in range(num_epochs):
            yield self.train_one_epoch()


if __name__ == "__main__":
    # Create a dummy model, data loader, and loss function
    model = nn.Linear(5, 3)
    dataloader = DataLoader([(torch.randn(5), torch.randn(3)) for _ in range(10)], batch_size=2)
    loss_fn = nn.MSELoss()

    # Create a TrainingLoop instance and train the model
    training_loop = TrainingLoop(model, dataloader, loss_fn, torch.device("cpu"))
    for epoch, loss in enumerate(training_loop.train(5)):
        print(f"Epoch {epoch+1}, Loss: {loss}")