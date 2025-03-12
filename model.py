# model.py
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)

def init_model(input_size=54, output_size=5, device='cpu'):
    model = DQN(input_size, output_size).to(device)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0.01)
    return model

def load_model_with_compatibility(model_path, input_size=54, output_size=5, device='cpu'):
    """Load a model with compatibility handling for architecture changes"""
    model = init_model(input_size, output_size, device)
    
    try:
        # Load the state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Check if it's a direct state dict or wrapped in a dictionary
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        # Filter out unexpected keys
        model_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        # Update the model dict with filtered state dict
        model_dict.update(filtered_state_dict)
        
        # Load the filtered state dict
        model.load_state_dict(model_dict, strict=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
        
    return model