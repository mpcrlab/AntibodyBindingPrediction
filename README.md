# Antibody Binding Site Prediction

A collection of code to pull proteins from uniprot, process them, train a CNN and LSTM, then evaluate their hidden weights for binding site prediction.

The corresponding data and model weights from the original study are included.

The pre-print paper can be found on [Biorxiv here](https://www.biorxiv.org/content/10.1101/2020.08.06.240101v1).

## Requirements:
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Installation:
- Please execute the following commands in a terminal.
```
git clone https://github.com/mpcrlab/AntibodyBindingPrediction.git
cd AntibodyBindingPrediction
pip3 install -r requirements.txt
cd ~
```

## How to use your own data:
0. Follow the installation steps above.
1. Please put your data in the format of 'Data_Folder/bind_class/bind_trainset.npy', 'Data_Folder/notbind/nobind_train.npy', 'Data_Folder/bind/bind_test.npy', '/notbind/nobind_test.npy'. The name of Data_Folder can be any, it is specified as an argument to the python file (step 3). The rest of the folders and files need to have these names. The numpy files should contain one sequence per row. The sequence needs to be represented as integers. 
2. ```cd AntibodyBindingPrediction```
3. Run the program with ```python3 Predict.py```. You will have specify the arguments for argparser, which can be interpreted with ``` python3 Predict.py --help```. 
