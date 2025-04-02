# evaluation.py
from sklearn.metrics import f1_score
import torch

def evaluate_model(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = (outputs[:, 1] > 0.5).int()  # Convert probabilities to binary labels
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    f1 = f1_score(y_true, y_pred)
    print(f"F1 Score: {f1:.3f}")
    return f1
