import torch
import torch.nn as nn
import torch.optim as optim
from net import HistoCRCNet  # 导入您的网络结构
from data.dataset import get_data_loaders

def evaluate_model_on_validation_set(model, validation_loader, device):
    model.eval()
    correct_preds = {classname: 0 for classname in range(9)}
    total_preds = {classname: 0 for classname in range(9)}
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_preds[label.item()] += 1
                total_preds[label.item()] += 1

    for classname, correct_count in correct_preds.items():
        accuracy = 100 * float(correct_count) / total_preds[classname]
        print(f'Accuracy for class {classname} is: {accuracy:.1f}%')

    model.train()
    return 100 * correct / total

def train_model(dataset_path, num_epochs=5):
    train_loader, validation_loader = get_data_loaders(dataset_path)
    model = HistoCRCNet(num_classes=9)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    best_acc = 0.0
    no_improve_count = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                running_loss = 0.0
                validation_accuracy = evaluate_model_on_validation_set(model, validation_loader, device)
                print(f'Epoch {epoch+1}, Validation Accuracy: {validation_accuracy}%')

                if validation_accuracy > best_acc:
                    best_acc = validation_accuracy
                    no_improve_count = 0
                    torch.save(model.state_dict(), 'best_model.pth')
                    print("Saved Better Model")
                else:
                    no_improve_count += 1

                if no_improve_count >= 15:
                    print("Stopping early due to no improvement in validation accuracy.")
                    break
        scheduler.step()

    print("Training Complete")
