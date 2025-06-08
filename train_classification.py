import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from app.utils.dataset import DepthDataset, create_data_loader
from app.utils.env import Env
from models.ClassificationCNN import ClassificationCNN
from models.DepthEstimationNetClass import DepthEstimationNetClass


class CustomOrdinalLoss(nn.Module):
    def __init__(self, n_classes = 80):
        super(CustomOrdinalLoss, self).__init__()
        self.num_bins = 80
        self.alpha = 1.0

        self.register_buffer('H', self._compute_information_gain_matrix())

    def _compute_information_gain_matrix(self):
        """
        Compute the B x B symmetric information gain matrix.
        H(p, q) = exp[-alpha * (p - q)^2]
        """
        # Create indices for all bin pairs
        p_indices = torch.arange(self.num_bins).float().unsqueeze(1)  # [B, 1]
        q_indices = torch.arange(self.num_bins).float().unsqueeze(0)  # [1, B]

        # Compute H(p, q) = exp[-alpha * (p - q)^2]
        H = torch.exp(-self.alpha * (p_indices - q_indices) ** 2)

        return H.to('cuda')

    def forward(self, logits, targets):
        """
        Forward pass of the loss function.

        Args:
            logits (torch.Tensor): Network output of shape [N, B, H, W] or [N, B]
                                 where N is batch size, B is number of bins
            targets (torch.Tensor): Ground truth depth labels of shape [N, H, W] or [N]
                                  with values in range [0, B-1] (0-indexed)

        Returns:
            torch.Tensor: Computed loss value
        """
        N, B, H = logits.shape
        logits = logits.permute(0, 2, 1).contiguous().view(-1, B)  # [N*H*W, B]
        targets = targets.view(-1)  # [N*H*W]

        # Ensure targets are long type for indexing
        targets = targets.long()

        # Compute probabilities P(D|z_i) using softmax
        log_probs = F.log_softmax(logits, dim=1)  # [N*pixels, B]

        # Get the information gain weights for each pixel
        # H[targets, :] gives us H(D*_i, D) for all D for each pixel i
        H_weights = self.H[targets]  # [N*pixels, B]

        # Compute weighted log probabilities
        weighted_log_probs = H_weights * log_probs  # [N*pixels, B]

        # Sum over all depth bins D for each pixel
        pixel_losses = torch.sum(weighted_log_probs, dim=1)  # [N*pixels]

        # Apply negative sign and reduction
        # if self.reduction == 'mean':
        loss = -pixel_losses.mean()
        # elif self.reduction == 'sum':
        #     loss = -pixel_losses.sum()
        # elif self.reduction == 'none':
        #     loss = -pixel_losses
        # else:
        #     raise ValueError(f"Unsupported reduction: {self.reduction}")

        return loss



class ClassificationDataset(DepthDataset):

    def __init__(self, root_dir, transform=None, target_transform=None, limit_cameras=False, num_classes=80, max_depth=80.0):
        super().__init__(root_dir, transform, target_transform, limit_cameras)
        self.num_classes = num_classes
        self.max_depth = max_depth

    def depth_to_class(self, depth_map):
        depth_map = np.clip(depth_map, 0, self.max_depth)
        return ((depth_map / self.max_depth) * (self.num_classes - 1)).astype(np.int64)

    def __getitem__(self, idx):
        image_path = self.raw_paths[idx]
        image = Image.open(image_path).convert('RGB')

        depth_path = self.target_paths[idx]
        depth = Image.open(depth_path)
        depth = np.array(depth).astype(np.float32) / 256

        depth_classes = self.depth_to_class(depth)

        if self.transform:
            image = self.transform(image)

        depth_classes = cv2.resize(depth_classes.astype(np.uint8), (848, 256), interpolation=cv2.INTER_NEAREST)
        depth_classes = torch.from_numpy(depth_classes).long()

        return image, depth_classes


def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')

        for images, depths in train_pbar:
            images, depths = images.to(device), depths.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            outputs = F.interpolate(outputs, size=depths.shape[-2:], mode='bilinear', align_corners=False)

            mask = depths[0, ] > 0
            loss = criterion(outputs[:, :, mask], depths[:, mask])

            # loss = criterion(outputs, depths)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        correct_pixels = 0
        total_pixels = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, depths in val_pbar:
                images, depths = images.to(device), depths.to(device)

                outputs = model(images)
                outputs = F.interpolate(outputs, size=depths.shape[-2:], mode='bilinear', align_corners=True)

                mask = depths[0, ] > 0
                loss = criterion(outputs[:, :, mask], depths[:, mask])

                # loss = criterion(outputs, depths)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                mask = depths > 0
                correct_pixels += ((predicted == depths) * mask).sum().item()
                total_pixels += mask.sum().item()

                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0

        scheduler.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'Pixel Accuracy: {pixel_accuracy:.4f}')
        print('-' * 50)

    output_path = Path('output')
    name = "ClassificationCNNCEL"

    if not output_path.exists():
        output_path.mkdir(exist_ok=True, parents=True)

    output_file_name = output_path / f"{name}.json"

    if output_file_name.exists():
        output_file_name.unlink()

    with open(output_file_name, "w") as f:
        json.dump({"train_loss": train_losses, "val_loss": val_losses}, f)

    onnx_path = output_path / f"{name}.onnx"

    if onnx_path.exists():
        onnx_path.unlink()

    model.eval()
    dummy_input = torch.randn(1, 3, 256, 848, requires_grad=True)
    dummy_input = dummy_input.to(device)
    torch.onnx.export(model, dummy_input, str(onnx_path.resolve()))

    return train_losses, val_losses


def main():
    BATCH_SIZE = 16
    NUM_EPOCHS = 40
    NUM_CLASSES = 80
    MAX_DEPTH = 80.0
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        # transforms.Resize((240, 320)),
        transforms.Resize((256, 848)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    env = Env()

    train_dataset = ClassificationDataset(
        root_dir=env.dataset_path / "train",
        transform=transform,
        limit_cameras=not env.both_cameras,
        num_classes=NUM_CLASSES,
        max_depth=MAX_DEPTH
    )

    val_dataset = ClassificationDataset(
        root_dir=env.dataset_path / "val",
        transform=transform,
        limit_cameras=not env.both_cameras,
        num_classes=NUM_CLASSES,
        max_depth=MAX_DEPTH
    )

    train_loader = create_data_loader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = create_data_loader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=4)

    # model = ClassificationCNN(num_classes=NUM_CLASSES).to(DEVICE)
    model = DepthEstimationNetClass(num_classes=NUM_CLASSES).to(DEVICE)

    print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')
    print(f'Training on {DEVICE}')

    train_model(
        model, train_loader, val_loader,
        num_epochs=NUM_EPOCHS, device=DEVICE
    )

if __name__ == '__main__':
    main()