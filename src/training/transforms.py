from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def _convert_to_rgb(image):
    return image.convert('RGB')

preprocess_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    _convert_to_rgb,
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
preprocess_val = transforms.Compose([
    transforms.CenterCrop(224),
    _convert_to_rgb,
    transforms.ToTensor(),
    normalize,
])