import os, sys, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Training.Model.FCNN import simpleDNN4l


def get_model_architecture(model_name, model_shape, rand_weights=True):

    if model_name.startswith('DNN'):
        return simpleDNN4l(model_shape, rand_weights=rand_weights)


def load_model(pth=None, layers=[784, 320, 120, 40, 10]):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if pth == None:
        pth = os.getcwd() + f"/Checkpoints/DNNandTMNIST_all_layers_run_0.pt"
        
    ckpt = torch.load(pth, weights_only=True)

    model = get_model_architecture('DNNandTMNIST', layers)
    model.load_state_dict(ckpt); model.to(device)

    return model