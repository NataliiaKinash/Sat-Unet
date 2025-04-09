import pickle
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
from unet import UNET
from utils import check_accuracy, get_loaders

IMAGE_HEIGHT = 128  # 256  # 128
IMAGE_WIDTH = 128  # 256  # 128

def evaluate_model_on_dataset(model_path, dataset_path):
    trained_model = UNET(in_channels=3, out_channels=1)
    trained_model.load_state_dict(torch.load(model_path))

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    train_data = data['train']
    val_data = data['val']
    
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    _, val_loader = get_loaders(
        train_data,
        val_data,
        batch_size=16,
        train_transform=None,
        val_transform=val_transforms,
    )
    check_accuracy(val_loader, trained_model, device="cpu", accuracy=True, dice=True, iou=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset.")

    parser.add_argument("model_path", type=str, help="Path to the trained model file")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset (pickle file)")

    args = parser.parse_args()

    evaluate_model_on_dataset(args.model_path, args.dataset_path)
    