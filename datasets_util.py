
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms

transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((384,384)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(size=(384,384), scale=(1-0.3, 1)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

def open_image_and_apply_transform(image_path):
    """Given the path of an image, open the image, and return it as a normalized tensor.
    """
    
    pil_image = Image.open(image_path)
    tensor_image = transform_test(pil_image)
    return tensor_image

def open_image_and_apply_transform_train(image_path):
    """Given the path of an image, open the image, and return it as a normalized tensor.
    """
    
    pil_image = Image.open(image_path)
    tensor_image = transform_train(pil_image)
    return tensor_image