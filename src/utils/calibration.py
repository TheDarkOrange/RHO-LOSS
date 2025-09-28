import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        return logits / self.temperature

def temperature_scale_model(model, val_loader, device):
    model.eval()
    model = model.to(device)

    logits_list = []
    labels_list = []

    with torch.no_grad():
        for _, inputs, targets in tqdm(val_loader, desc="Collecting logits"):
            inputs = inputs.to(device)
            logits = model(inputs)
            logits_list.append(logits)
            labels_list.append(targets.to(device))

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    model_temp = ModelWithTemperature(model).to(device)

    optimizer = torch.optim.LBFGS([model_temp.temperature], lr=0.01, max_iter=50)

    def eval():
        optimizer.zero_grad()
        loss = F.cross_entropy(model_temp.temperature_scale(logits), labels)
        loss.backward()
        return loss

    optimizer.step(eval)

    return model_temp.temperature.item()
