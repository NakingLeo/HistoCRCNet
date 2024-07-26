import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from models.regression import ModifiedResNet101

def load_and_transform_image(image_path, target_size=256):
    image = Image.open(image_path)
    width, height = image.size
    scale = target_size / min(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height), Image.ANTIALIAS)
    padding_left = (target_size - new_width) // 2
    padding_right = target_size - new_width - padding_left
    padding_top = (target_size - new_height) // 2
    padding_bottom = target_size - new_height - padding_top
    image = ImageOps.expand(image, border=(padding_left, padding_top, padding_right, padding_bottom), fill=0)
    return image

def predict_image(image_path, model, device, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image).item()
    return output

def update_filename(image_path, hazard_pred):
    filename, extension = os.path.splitext(image_path)
    new_filename = f"{filename}_{hazard_pred:.2f}{extension}"
    os.rename(image_path, new_filename)
    return new_filename

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModifiedResNet101().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    folder = 'G:/hospital/validate/validateSet'
    for file in os.listdir(folder):
        image_path = os.path.join(folder, file)
        hazard_pred = predict_image(image_path, model, device, transform)
        print(hazard_pred)
        new_image_path = update_filename(image_path, hazard_pred)
        print(f"Updated filename: {new_image_path}")

if __name__ == "__main__":
    main()
