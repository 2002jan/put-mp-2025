import matplotlib.pyplot as plt
from torchvision import transforms

from utils.dataset import DepthDataset, create_data_loader

if __name__ == "__main__":
    image_size = 256
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    target_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    dataset = DepthDataset(
        root_dir="F:\\kitti\\data\\val",
        transform=transform,
        target_transform=target_transform,
    )

    dataloader = create_data_loader(
        dataset,
        batch_size=16,
        num_workers=4,
        shuffle=True
    )

    for i, (raw_images, target_images) in enumerate(dataloader):
        print(f"Batch {i}: Raw shape: {raw_images.shape}, Target shape: {target_images.shape}")

        fig = plt.figure()

        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(raw_images[0].permute(1, 2, 0))

        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(target_images[0].permute(1, 2, 0))
        plt.show(block=True)

        if i >= 2:
            break
