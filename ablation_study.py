import torch.nn as nn

# ablation_study.py

from editguard import get_model

def ablation_experiment(remove_component=None):
    model = get_model()
    
    if remove_component == "BO":
        model.fc1 = nn.Identity()  # Remove one layer as an example

    print(f"Running ablation study - Removing {remove_component}")
    return model
