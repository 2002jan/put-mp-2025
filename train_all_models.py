import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from app.utils.env import Env
#from models.DepthEstimationNet import DepthEstimationNet
#from models.SmallCNN import SmallCNN
#from models.UNet import UNet
from models.UNetAlikeCNN import UNetAlikeCNN
from models.UNetAlikeDeeperCNN import UNetAlikeDeeperCNN
#from models.UNetResNetDepth import UNetResNetDepth
from app.utils.dataset import DepthDataset, create_data_loader
from torchvision import transforms
import torch.optim as optim
import gc
import numpy as np

from models.train_depth_model import SILogLoss, train_one_epoch, evaluate


def check_for_nan_inf(tensor, name="tensor"):
    """Check tensor for NaN or Inf values"""
    if torch.isnan(tensor).any():
        print(f'NaN detected in {name}')
        return True
    if torch.isinf(tensor).any():
        print(f'Inf detected in {name}')
        return True
    return False


def contains_nan(model):
    """Check if model parameters contain NaN"""
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f'NaN detected in parameter: {name}')
            return True
    return False


def check_gradients(model, max_grad_norm=10.0):
    """Monitor gradient norms and detect issues"""
    total_norm = 0
    nan_grads = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2)
            total_norm += grad_norm.item() ** 2

            if torch.isnan(grad_norm):
                nan_grads.append(name)
            elif grad_norm > max_grad_norm:
                print(f'Large gradient in {name}: {grad_norm:.4f}')

    total_norm = total_norm ** (1. / 2)

    if nan_grads:
        print(f'NaN gradients detected in: {nan_grads}')
        return True, total_norm

    return False, total_norm


def init_weights(m):
    """Proper weight initialization"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def get_optimal_batch_size(name: str, train_ds: DepthDataset, val_ds: DepthDataset) -> int:
    if not torch.cuda.is_available():
        return 16

    batch_size = 0

    device = torch.device("cuda")

    first_run = True

    while True:
        batch_size += 16

        model = globals()[name]()
        train_loader = create_data_loader(train_ds, batch_size=batch_size)
        val_loader = create_data_loader(val_ds, batch_size=batch_size, shuffle=False)

        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        loss_fn = SILogLoss()

        model.train()

        for images, depths in tqdm(train_loader, desc="Training"):
            images, depths = images.to(device), depths.to(device)
            optimizer.zero_grad()
            preds = model(images)

            if preds.shape[-2:] != depths.shape[-2:]:
                preds = nn.functional.interpolate(preds, size=depths.shape[-2:], mode="bilinear", align_corners=True)

            loss = loss_fn(preds.squeeze(1), depths.squeeze(1))
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            for images, depths in tqdm(val_loader, desc="Validation"):
                images, depths = images.to(device), depths.to(device)
                preds = model(images)

                if preds.shape[-2:] != depths.shape[-2:]:
                    preds = nn.functional.interpolate(preds, size=depths.shape[-2:], mode="bilinear",
                                                      align_corners=True)

                loss = loss_fn(preds.squeeze(1), depths.squeeze(1))

        free = torch.cuda.mem_get_info()[0] / 1024 ** 3
        total = torch.cuda.mem_get_info()[1] / 1024 ** 3
        total_cubes = 24
        free_cubes = int(total_cubes * free / total)
        print(f'VRAM: {total - free:.2f}/{total:.2f}GB\t VRAM:[' + (
                total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']')

        del optimizer
        del model
        gc.collect()
        torch.cuda.empty_cache()

        if not first_run:
            break

        if first_run:
            batch_size *= math.ceil(total / (total - free))
            batch_size = int(batch_size)
            first_run = False

        if free < 0:
            batch_size -= 16
            if batch_size < 0:
                batch_size = 16

            break

    return batch_size


def train_model(name: str, env: Env, train_ds: DepthDataset, val_ds: DepthDataset):
    print(f"Testing optimal batch size for {name}")
    bs = get_optimal_batch_size(name, train_ds, val_ds)
    # bs = 16
    print(f"Batch size found: {bs}")

    print(f"Training model: {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = globals()[name]()

    model.apply(init_weights)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3 if "UNetAlikeDeeperCNN" not in name else 1e-4, eps=1e-8, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    loss_fn = SILogLoss()

    train_losses = []
    val_losses = []

    best_val_loss = None
    best_val_epoch = 0

    train_loader = create_data_loader(train_ds, batch_size=bs)
    val_loader = create_data_loader(val_ds, batch_size=bs, shuffle=False)

    num_epochs = 400

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        print('-' * 50)
        # train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        # val_loss = evaluate(model, val_loader, loss_fn, device)

        model.train()
        running_loss = 0.0
        nan_detected = False

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            try:
                output = model(data)

                if check_for_nan_inf(output, f"model output (batch {batch_idx})"):
                    print("NaN/Inf detected in forward pass! Skipping batch.")
                    continue

                loss = loss_fn(output, target)

                if check_for_nan_inf(loss, f"loss (batch {batch_idx})"):
                    print("NaN/Inf detected in loss! Skipping batch.")
                    continue

                loss.backward()

                has_nan_grad, grad_norm = check_gradients(model)
                if has_nan_grad:
                    print("NaN gradients detected! Skipping optimization step.")
                    optimizer.zero_grad()  # Clear bad gradients
                    continue

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Check model parameters for NaN after update
                if contains_nan(model):
                    print("NaN detected in model parameters after optimization!")
                    nan_detected = True
                    break

                running_loss += loss.item()

                if batch_idx % 20 == 0:
                    print(f'Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.6f}, '
                          f'Grad Norm: {grad_norm:.4f}, '
                          f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        if nan_detected:
            print("Training stopped due to NaN detection!")
            break

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)

                output = model(data)

                # Check validation outputs
                if check_for_nan_inf(output, "validation output"):
                    print("NaN/Inf detected in validation!")
                    break

                val_loss += loss_fn(output, target).item()

                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Train Loss: {avg_train_loss:.6f}')
        print(f'Val Loss: {avg_val_loss:.6f}')
        print(f"Epochs till early stopping {30 - (epoch - best_val_epoch)}")

        scheduler.step(avg_val_loss)

        if avg_val_loss > 10.0 or np.isnan(avg_val_loss):
            print("Training stopped due to exploding validation loss!")
            break

        # scheduler.step(val_loss)
        #
        # train_losses.append(train_loss)
        # val_losses.append(val_loss)

        # print(
        #     f"Train Loss: {train_loss} | Val Loss: {val_loss} | Learning rate: {optimizer.param_groups[0]['lr']} | Epochs till early stopping {30 - (epoch - best_val_epoch)}")

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch

        if epoch - best_val_epoch > 30:
            print("Early stopping triggered")
            break

    output_path = Path('output')

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

    del optimizer
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    models = [m for m in globals().keys() if "net" in m.lower() or "cnn" in m.lower()]

    env = Env()

    transform = transforms.Compose([
        transforms.Resize((256, 848)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    target_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 848)),
        transforms.ToTensor()
    ])

    train_dataset = DepthDataset(
        root_dir=env.dataset_path / "train",
        transform=transform,
        target_transform=target_transform,
        limit_cameras=not env.both_cameras
    )
    val_dataset = DepthDataset(
        root_dir=env.dataset_path / "val",
        transform=transform,
        target_transform=target_transform,
        limit_cameras=not env.both_cameras
    )

    for model in models:
        train_model(model, env, train_dataset, val_dataset)


if __name__ == "__main__":
    main()
