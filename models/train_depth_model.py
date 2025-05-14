import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic Loss for depth estimation.
    """
    def __init__(self, variance_focus=0.85):
        super().__init__()
        self.variance_focus = variance_focus

    def forward(self, pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1)

        mask = target > 0
        masked_pred = pred[mask]
        masked_target = target[mask]

        log_diff = torch.log(masked_pred) - torch.log(masked_target)

        silog1 = torch.mean(log_diff ** 2)
        silog2 = torch.mean(log_diff) ** 2
        return silog1 - silog2


        # mask = target > 0  # valid depth mask
        # pred = pred[mask]
        # target = target[mask]
        # log_diff = torch.log(pred) - torch.log(target)
        # silog = torch.mean(log_diff ** 2) - self.variance_focus * torch.mean(log_diff) ** 2
        # return silog

class BerHuLoss(nn.Module):
    """
    BerHu (Reverse Huber) Loss for depth estimation.
    Combines L1 and L2 loss depending on threshold c.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        mask = target > 0  # valid depth mask
        pred = pred[mask]
        target = target[mask]
        error = torch.abs(pred - target)
        c = 0.2 * torch.max(error).item()  # threshold can be adjusted

        l1_part = error[error <= c]
        l2_part = error[error > c]

        loss = 0.0
        if len(l1_part) > 0:
            loss += torch.sum(l1_part)
        if len(l2_part) > 0:
            loss += torch.sum((l2_part ** 2 + c ** 2) / (2 * c))

        return loss / error.numel()


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0

    for images, depths in tqdm(dataloader, desc="Training"):
        images, depths = images.to(device), depths.to(device)
        optimizer.zero_grad()
        preds = model(images)

        # Resize prediction to match ground truth shape if needed
        if preds.shape[-2:] != depths.shape[-2:]:
            preds = nn.functional.interpolate(preds, size=depths.shape[-2:], mode="bilinear", align_corners=True)

        loss = loss_fn(preds.squeeze(1), depths.squeeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, depths in tqdm(dataloader, desc="Validation"):
            images, depths = images.to(device), depths.to(device)
            preds = model(images)

            if preds.shape[-2:] != depths.shape[-2:]:
                preds = nn.functional.interpolate(preds, size=depths.shape[-2:], mode="bilinear", align_corners=True)

            loss = loss_fn(preds.squeeze(1), depths.squeeze(1))
            total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)


def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4, device='cuda'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=6, verbose=True)

    loss_fn = SILogLoss()

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss} | Val Loss: {val_loss}")

    # write losses to file separate by comma
    with open('train_losses.txt', 'w') as f:
        for i in range(num_epochs):
            f.write(f"{train_losses[i]},")   
        
    with open('val_losses.txt', 'w') as f:
        for i in range(num_epochs):
            f.write(f"{val_losses[i]},")
    # write losses to file separate by comma

    # Plotting the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Reverse huber Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()