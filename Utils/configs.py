conf = {
    "simpleDNNTMnist": {
        "model_name": "DNNandTMNIST",
        "model_layers": [784, 400, 100],
        "num_classes": 10,
        "dataset": "typeface_mnist",
        "batch_size": 64,
        "epochs": 10,
        "lr": 0.01,
        "optimizer": "sgd", 
        "scheduler": "simple/cycleLR",
        "baseline_val": 0.6
    },

    "simpleDNNTMnistTransf": {
        "model_name": "DNNandTransfTMNIST",
        "model_layers": [784, 512, 256, 100],
        "num_classes": 10,
        "dataset": "typeface_mnist_augment",
        "batch_size": 64,
        "lr": 0.01,
        "epochs": 10,
        "optimizer": "sgd/adamW", 
        "scheduler": "simple/cycleLR",
        "baseline_val": 0.6
    },

    "simpleDNNcorruptedMnist": {
        "model_name": "DNNandCorruptMNIST",
        "model_layers": [1024, 400, 100],
        "num_classes": 10,
        "dataset": "corrupted_mnist",
        "batch_size": 64,
        "lr": 0.0005,
        "epochs": 10,
        "optimizer": "sgd/adamW", 
        "scheduler": "simple/cycleLR",
        "baseline_val": 0.4
    },
    
    "simpleDNNaffineMnist": {
        "model_name": "DNNandAffineMNIST",
        "model_layers": [784, 512, 256, 128, 64],
        "num_classes": 10,
        "dataset": "affine_mnist",
        "batch_size": 64,
        "lr": 0.05,
        "epochs": 10,
        "optimizer": "sgd", 
        "scheduler": "simple/cycleLR",
        "baseline_val": 0.4
    },

    "simpleDNNPAM50": {
        "model_name": "DNNandPAM50",
        "model_layers": [50, 32, 8],
        "num_classes": 2,
        "dataset": "pam50",
        "batch_size": 64,
        "lr": 0.01,
        "epochs": 8,
        "optimizer": "sgd/adamW", 
        "scheduler": "simple/cycleLR",
    }
}