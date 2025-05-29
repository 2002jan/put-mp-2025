import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from app.utils.env import Env
from models.DepthEstimationNet import DepthEstimationNet
from models.SmallCNN import SmallCNN
from models.UNet import UNet
from models.UNetAlikeCNN import UNetAlikeCNN
from models.UNetAlikeDeeperCNN import UNetAlikeDeeperCNN
from models.UNetResNetDepth import UNetResNetDepth
from app.utils.dataset import DepthDataset, create_data_loader
from torchvision import transforms
import torch.optim as optim
import gc

from models.train_depth_model import SILogLoss, train_one_epoch, evaluate


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

        if first_run:
            batch_size *= math.ceil(total / free)
            first_run = False
            continue

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
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=9, verbose=True)

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
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss} | Val Loss: {val_loss}")

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
