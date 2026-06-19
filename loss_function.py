import torch
import torch.nn as nn

class CrossEntropyLossFunction(nn.Module):
    """
    A PyTorch module that computes the cross-entropy loss between the model's output and the target labels.
    """

    def __init__(self, reduction: str = 'mean'):
        """
        Initializes the CrossEntropyLossFunction module.

        Args:
        reduction (str): The reduction method to use when computing the loss. Defaults to 'mean'.
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the cross-entropy loss between the model's output and the target labels.

        Args:
        predictions (torch.Tensor): The model's output.
        targets (torch.Tensor): The target labels.

        Returns:
        torch.Tensor: The computed cross-entropy loss.
        """
        return self.criterion(predictions, targets)


if __name__ == "__main__":
    # Create a dummy model output and target labels
    predictions = torch.randn(10, 5)
    targets = torch.randint(0, 5, (10,))

    # Create an instance of the CrossEntropyLossFunction module
    loss_function = CrossEntropyLossFunction()

    # Compute the cross-entropy loss
    loss = loss_function(predictions, targets)

    # Print the computed loss
    print(loss)