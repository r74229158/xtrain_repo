
### Running the Code 
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
publicly available on Kaggle
The main python file for running the code is `run.py`. Parameters related to training are controlled by predefined configurations in [`configs.py`](./Utils/configs.py). They can be adjusted according to one's needs. Researchers can select from a range of available datasets: Typeface MNIST, CorruptedMNIST (both publicly available on Kaggle) and affNIST (publicly available on the webpage of the University of Toronto). Each is being automatically downloaded when needed.

> The PAM50 dataset requires a lengthy procedure for acquiring the data and is described on the paper. 

Run the code as:

```bash
python run.py 
```

to select a different dataset (rather than TMNIST) and a different number of data to track Xtrain run:
```bash
python run.py -conf simpleDNNcorruptedMnist -nt 20
```

Results are displayed in a subfolder under `/Results`, according to the name of the dataset being used. Experiments presented in Supplementary Material can be found in ipynb files in `Theory`.