import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from net import HistoCRCNet

def predict_img(source_dir):
    classes = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HistoCRCNet(num_classes=9)
    model.load_state_dict(torch.load('best_model.pth'))
    model = model.to(device)
    model.eval()

    for cls in classes:
        os.makedirs(os.path.join(source_dir, cls), exist_ok=True)

    for img_name in os.listdir(source_dir):
        img_path = os.path.join(source_dir, img_name)
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert('RGB')
            img_transformed = transform(img)
            img_transformed = img_transformed.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_transformed)
                _, predicted = torch.max(output, 1)
                class_name = classes[predicted.item()]

            new_img_path = os.path.join(source_dir, class_name, img_name)
            os.rename(img_path, new_img_path)

    print("All images have been processed and moved to their respective category folders.")
