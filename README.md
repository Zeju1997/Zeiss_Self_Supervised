# Zeiss_Self_Supervised

#### Step 1: Create Conda environment

```bash
conda create -n vissl_env python=3.8
source activate vissl_env
```

#### Step 2: Install PyTorch

```bash
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python
pip install -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/py38_cu102_pyt181/download.html apex
```

#### Step 3: Install VISSL

```bash
pip install vissl
# verify installation
python -c 'import vissl'
```

#### Step 3: Install VISSL

```bash
pip install vissl
# verify installation
python -c 'import vissl'
```

#### Step 4: Install Jupyter Notebook

```bash
conda install jupyter                # install jupyter + notebook
jupyter notebook
```

#### Step 4: Install Some Modules

```bash
pip install tensorboard
conda install pandas
conda install matplotlib
pip install scikit-image
```
