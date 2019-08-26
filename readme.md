# Purpose
    "Detect abnormal traffic by one class classfication"  

## 1. Requirements
    python 3.x
    pytorch 0.4.1
    sklearn 0.20.0


## 2. Project Directory Structure
### |- input_data: raw data
    if any data is more than 100MB, please do not store it at here
    data/Wednesday-workingHours-withoutInfinity-Sampled.pcap_ISCX.csv

### |- output_data: results
    ...
    
### |- log: use to log middle or tmp results.
    ...
    
### |- proposed_algorithms
    ### |- deep_autoencoder_pytorch
            main_autoencoder.py
    each folder just includes one algorithm

### |- compared_algorithms
    ### |- OCSVM_Sklearn
        main_ocsvm.py
        basic_svm.py

### |- utilities
    CSV_Dataloder.py
    common_funcs.py
        CSV_Dataloder.py
    
    ## 'pcap2flow' folder
    >>>--- toolkit to convert pcap files to txt or feature data.
    
    ## 'preprocess' folder 
    >>>--- toolkit to preprocess input data, such as 'load data', 'normalization data'
        
    ## |- visualization: plot data to visualize 
        ..

### |-history_files: backup 
    ...

## Note:
    In terms of the accuracy of an IDS, there are four possible states for each activity observed. A true positive state 
    is when the IDS identifies an activity as an attack and the activity is actually an attack. A true positive is a 
    successful identification of an attack. A true negative state is similar. This is when the IDS identifies an activity 
    as acceptable behavior and the activity is actually acceptable. A true negative is successfully ignoring acceptable 
    behavior. Neither of these states are harmful as the IDS is performing as expected. A false positive state is when the
    IDS identifies an activity as an attack but the activity is acceptable behavior. A false positive is a false alarm. 
    A false negative state is the most serious and dangerous state. This is when the IDS identifies an activity as 
    acceptable when the activity is actually an attack. That is, a false negative is when the IDS fails to catch an attack. 
    This is the most dangerous state since the security professional has no idea that an attack took place. False positives, 
    on the other hand, are an inconvenience at best and can cause significant issues. However, with the right amount of overhead, 
    false positives can be successfully adjudicated; false negatives cannot. 
    
    Concept in IDS 
        Positive: attack or anormaly events; Negative: normal events.
        There are 4 main types of IDS alerts. These are : 
            True Positive  (TP): Bad traffic which triggers an alert. 
            False Positive (FP): Good traffic which triggers an alert.
            False Negative (FN): Bad traffic, but no alert is raised.
            True Negative  (TN): Good traffic, and no alert is raised.
    
    reference: 
        https://www.owasp.org/index.php/Intrusion_Detection