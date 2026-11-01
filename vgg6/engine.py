"""One epoch of training and a full pass of evaluation."""
import torch
import torch.nn.functional as F


def train_one_epoch(model, loader, optimizer, device, on_step=None):
    """Trains for one epoch. Calls on_step(batch_loss, batch_acc) after every batch."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        batch_correct = (out.argmax(1) == y).sum().item()
        total_loss += loss.item() * y.size(0)
        correct += batch_correct
        seen += y.size(0)
        if on_step:
            on_step(loss.item(), batch_correct / y.size(0))
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns (mean loss, top-1 accuracy) over the loader."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        total_loss += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        seen += y.size(0)
    return total_loss / seen, correct / seen
