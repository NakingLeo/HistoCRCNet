import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps
from models.regression import ModifiedResNet101, CoxLoss, accuracy_cox, cox_log_rank, CIndex_lifeline

class ResizeAndPad:
    def __init__(self, output_size=(256, 256)):
        self.output_size = output_size

    def __call__(self, img):
        left = (self.output_size[0] - img.size[0]) // 2
        right = self.output_size[0] - img.size[0] - left
        top = (self.output_size[1] - img.size[1]) // 2
        bottom = self.output_size[1] - img.size[1] - top
        img = transforms.Pad((left, top, right, bottom), fill=0, padding_mode='constant')(img)
        return img

class SurvivalDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.filenames = os.listdir(directory)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        parts = filename.split('_')
        survtime = float(parts[-2])
        censor = float(parts[-1].split('.')[0])
        image_path = os.path.join(self.directory, filename)
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, survtime, censor

def load_data(dataset, batch_size=64, train_split=0.8):
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    risk_pred_all = np.array([])
    censor_all = np.array([])
    survtime_all = np.array([])

    with torch.no_grad():
        for images, survtimes, censors in val_loader:
            images = images.to(device)
            survtimes = survtimes.to(device)
            censors = censors.to(device)
            hazard_pred = model(images).squeeze()
            loss = loss_fn(survtime=survtimes, censor=censors, hazard_pred=hazard_pred, device=device, model=model)
            total_loss += loss.item()

            hazard_pred_array = hazard_pred.detach().cpu().numpy()
            risk_pred_all = np.concatenate((risk_pred_all, hazard_pred_array))
            censor_all = np.concatenate((censor_all, censors.cpu().numpy()))
            survtime_all = np.concatenate((survtime_all, survtimes.cpu().numpy()))

    acc_cox = accuracy_cox(risk_pred_all, censor_all)
    p_value = cox_log_rank(risk_pred_all, censor_all, survtime_all)
    c_index = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)

    return total_loss / len(val_loader), acc_cox, p_value, c_index

def train_model(dataset_path, num_epochs=30):
    transform = transforms.Compose([
        ResizeAndPad((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

    dataset = SurvivalDataset(dataset_path, transform=transform)
    train_loader, val_loader = load_data(dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModifiedResNet101().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    loss_fn = CoxLoss

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])

        for images, survtimes, censors in train_loader:
            images = images.to(device)
            survtimes = survtimes.to(device)
            censors = censors.to(device)

            optimizer.zero_grad()
            hazard_pred = model(images).squeeze()
            loss = CoxLoss(survtime=survtimes, censor=censors, hazard_pred=hazard_pred, device=device, model=model)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            hazard_pred_array = hazard_pred.detach().cpu().numpy()
            if hazard_pred_array.ndim == 0:
                hazard_pred_array = np.expand_dims(hazard_pred_array, 0)
            risk_pred_all = np.concatenate((risk_pred_all, hazard_pred_array))
            censor_all = np.concatenate((censor_all, censors.cpu().numpy()))
            survtime_all = np.concatenate((survtime_all, survtimes.cpu().numpy()))

        acc_cox = accuracy_cox(risk_pred_all, censor_all)
        p_value = cox_log_rank(risk_pred_all, censor_all, survtime_all)
        c_index = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
        print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader)}, Accuracy: {acc_cox:.4f}, P-value: {p_value:.6f}, C-index: {c_index:.4f}")

        val_loss, acc_cox, p_value, c_index = validate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}, Accuracy: {acc_cox:.4f}, P-value: {p_value:.4f}, C-index: {c_index:.4f}")

        scheduler.step(val_loss)
        if c_index > 0.72:
            model_filename = f'best_model_{p_value:.6f}_{c_index:.4f}.pth'
            torch.save(model.state_dict(), model_filename)
            print('Saved model')
