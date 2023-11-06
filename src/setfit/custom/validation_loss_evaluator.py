import logging
import torch

from torch import nn
from torch.utils.data import DataLoader
from torch import Tensor, device

from sentence_transformers.evaluation import SentenceEvaluator

logger = logging.getLogger(__name__)


def batch_to_device(batch: dict, target_device: device = 'cpu') -> dict:
    """
    Map a pytorch data batch to a device (cpu/gpu), as implemented in sentence-transformers.

    Args:
        batch: Dictionary containing `input_ids` and `attention_mask` tensors.
        target_device: The type of Torch device available (cuda or cpu).

    Returns: The given batch with tensor values mapped to the specified torch device.
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)

    return batch


class ValidationLossEvaluator(SentenceEvaluator):
    """Evaluator for validation loss values."""

    def __init__(self, dataloader: DataLoader, loss_model: nn.Module = None) -> None:
        """
        Initialize a ValidationLossEvaluator.

        Args:
            dataloader: The Data loader object.
            loss_model: The type of loss function.
        """
        self.dataloader = dataloader
        self.loss_model = loss_model
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        loss_model.to(self.device)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """
        This is called during training to evaluate the model.
        It returns the validation loss.
        """
        # Set behavior to evaluation (in case there's a different behavior than training)
        model.eval()
        self.loss_model.eval()

        # Initialize cumulative loss for this epoch
        cumulative_loss_value = 0

        # Apply batching
        self.dataloader.collate_fn = model.smart_batching_collate
        for step, batch in enumerate(self.dataloader):
            features, labels = batch
            # Set data to the correct device
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], self.device)
            labels = labels.to(self.device)
            with torch.no_grad():  # do not perform backprop (i.e., do not train)
                # Compute loss and cumulate it
                loss = self.loss_model(features, labels).item()
                cumulative_loss_value += loss

        # Compute average epoch loss
        epoch_loss = cumulative_loss_value / len(self.dataloader)

        return epoch_loss
