# Purpose
    "Detect abnormal traffic by one class classfication"  

## 1. Requirements
    python 3.x
    pytorch 0.4.1
    sklearn 0.20.0


## 2. Project Directory Structure
### |- Data
    data/Wednesday-workingHours-withoutInfinity-Sampled.pcap_ISCX.csv

### |- DeepAutoEncoder_Pytorch
    main_autoencoder.py

### |- OCSVM_Sklearn
    main_ocsvm.py
    basic_svm.py

### |- Utilities
    CSV_Dataloder.py
    common_funcs.py

## Note:
    since 10/13, we focus on case3, please read the codes related to case3.