"""Lazy torchvision loaders for the ResNet-18 CIFAR pilot."""

def cifar100(root: str, *, train: bool, download: bool = False):
    try:
        from torchvision import datasets, transforms
    except ImportError as error:
        raise RuntimeError("install the 'vision' extra to load CIFAR-100") from error
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ]) if train else transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    return datasets.CIFAR100(root, train=train, download=download,
                             transform=transform)

