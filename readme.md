# Purpose
    "Detect abnormal traffic by one class classfication"  

## 1. Requirements
    python 3.x
    pytorch 0.4.1
    sklearn 0.20.0


## 2. Project Directory Structure
### |- input_data: raw data
    data/Wednesday-workingHours-withoutInfinity-Sampled.pcap_ISCX.csv

### |- output_data: results

### |- log: use to log middle or tmp results.

### |- proposed_algorithms
    ### |- deep_autoencoder_pytorch
            main_autoencoder.py
### |- compared_algorithms
    ### |- OCSVM_Sklearn
        main_ocsvm.py
        basic_svm.py

### |- utilities
    CSV_Dataloder.py
    common_funcs.py

### |-history_files: backup 

## Note:
    since 10/13, we focus on case3, please read the codes related to case3.