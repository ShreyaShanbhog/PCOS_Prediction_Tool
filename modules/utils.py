import os
import random
import matplotlib.pyplot as plt
from collections import Counter


def check_class_balance(dataset, name="Dataset"):
    targets = dataset.targets
    class_counts = Counter(targets)

    print(f"\n{name} Class Distribution:")
    for class_index, count in class_counts.items():
        class_name = dataset.classes[class_index]
        print(f"{class_name}: {count} images")


def unnormalize(img, mean, std):
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img


def visualize_samples(
    dataset,
    mean,
    std,
    num_images=6,
    save_path="outputs/sample_visualization.png",
    show=False
):
    """
    Visualize and optionally save sample images from dataset.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(12, 6))

    for i in range(num_images):
        image, label = dataset[random.randint(0, len(dataset)-1)]
        image = unnormalize(image, mean, std)
        image = image.permute(1, 2, 0)

        plt.subplot(2, num_images // 2, i + 1)
        plt.imshow(image)
        plt.title(dataset.classes[label])
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()  # VERY IMPORTANT (prevents memory leaks)

    print(f"\nSaved visualization to: {save_path}")
