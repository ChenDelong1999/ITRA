from torchvision import transforms


def _convert_to_rgb(image):
    return image.convert('RGB')

def get_preprocess(image_resolution=224, is_train=False, aug=None):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        preprocess_train = transforms.Compose([
            transforms.RandomResizedCrop(image_resolution, scale=(0.9, 1.0)),
            _convert_to_rgb,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        return preprocess_train
    else:
        preprocess_val = transforms.Compose([
            transforms.CenterCrop(image_resolution),
            _convert_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])
        return preprocess_val

