# pcostraj/scripts/train_loader_check.py

from modules.dataset import get_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from modules.utils import check_class_balance, visualize_samples


if __name__ == "__main__":

    train_dir = "data/train"
    test_dir = "data/test"

    train_ds, test_ds, train_loader, test_loader = get_dataloaders(
        train_dir,
        test_dir,
        batch_size=16
    )

    print(f"\nTraining Images: {len(train_ds)}")
    print(f"Testing Images: {len(test_ds)}")
    print(f"Classes: {train_ds.classes}")

    check_class_balance(train_ds, "Training")
    check_class_balance(test_ds, "Testing")

    visualize_samples(train_ds, IMAGENET_MEAN, IMAGENET_STD)

    images, labels = next(iter(train_loader))
    print(f"\nBatch Shape: {images.shape}")
