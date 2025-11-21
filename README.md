## XtrAIn: Capturing Causal Effects during Training updates to calculate Feature Importance in Deep Neural Networks ꧁⎝ 𓆩༺✧༻𓆪 ⎠꧂

Welcome to the official repository for XtrAIn, an attribution method presented in CVPR 2026. What follows is a brief summary of XtrAIn's theory, along with a guide for running the code.

### Theoretical Aspects 📜
---

This is the main repository for XtrAIn (presented in CVPR 2026). It is a new attrbution method founded on the framework of **active learning**, which is based on explaining updates during the training process. Xtrain operates by *capturing the causal effects of changes* during training updates, as defined by equations

$$ df_i(x) = f_{W_{i} = W'_{i}}(x) - f(x)$$
$$ df'_i(x) = f'(x) - f'_{W'_{i} = W_{i}}(x)$$

and 

$$d\mathcal{R}^t(x) = \mathcal{I}^T \cdot (df(x) + df'(x))$$

The final score is calculated as 

$$ \mathcal{R}^{t+1}(x) = \mathcal{R}^t(x) + d\mathcal{R}^t(x)$$,
$$ \mathcal{R}^0(x) = 0. $$

<br>

### Running the Code 🤖
---
First, download all necessary dependencies:

```bash
# Create a new conda environment with Python
conda create -n myenv python=3.9 -y

# Activate the environment
conda activate myenv

# Install packages from requirements.txt
pip install -r requirements.txt
```

The main python file for running the code is `run.py`. Yet we advocate for the revision of the [`configs.py`](./Utils/configs.py) file according to your needs. Everything related to the training process is defined in this file. Researchers can select from a range of available datasets: [Typeface MNIST](https://www.kaggle.com/datasets/nimishmagre/tmnist-typeface-mnist), [affNIST](https://www.cs.toronto.edu/~tijmen/affNIST/) and [CorruptedMNIST](https://www.kaggle.com/datasets/shreyasi2002/corrupted-mnist). 

> The PAM50 dataset requires a lengthy procedure for acquiring the data and is described on the paper. 

Data will be automatically downloaded to the correct direction. Run the code as:

```bash
python run.py 
```

to select a different dataset (rather than TMNIST) and a different number of data to track XtrAIn run:
```bash
python run.py -conf simpleDNNcorruptedMnist -nt 20
```

Feel free to add any dataset you like and experiment with XtrAIn. To achieve this, pass the data in a folder in `Training/Data/Datasets` and define a `torch.utils.data.Dataset` class in `Training/Data/Modules`. Then pass it to  `Training/Data/Modules/custom_loader` to use it.
