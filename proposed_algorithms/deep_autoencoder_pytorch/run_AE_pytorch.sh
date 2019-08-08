#!/bin/bash
# the above line tells the shell how to execute this script
#
# job-name
#SBATCH --job-name=AE_pytorch
#
# need 4 nodes
##SBATCH --nodes=4 # if you write the code based on tensor, maybe you can use
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#
# expect the job to finish within 5 hours. If it takes longer than 5 hours, SLURM can kill it
#SBATCH --time=60:00:00
#
# expect the job to use no more than 24GB of memory
#SBATCH --mem=60GB
#
# once job ends, send me an email
#SBATCH --mail-type=END
#SBATCH --mail-user=kun.bj@icloud.com
#
# both standard output and error are directed to the same file.
#SBATCH --output=outlog_%A_%a.out
##SBATCH --error=_%A_%a.err
#SBATCH  --error=errlog_%A_%a.out
#
# first we ensure a clean running environment:
module purge
mkdir -p py3.6.3
# and load the module for the software we are using:
module load python3/intel/3.6.3
# create the virtual environment for install new libraries which do not need sudo permissions right.
virtualenv --system-site-packages py3.6.3
source py3.6.3/bin/activate
pip3 install pillow
pip3 install scapy
pip3 install torch
pip3 install sklearn
pip3 install matplotlib

#source py3.6.3/bin/activate /home/ky13/py3.6.3
cd /scratch/ky13/Experiments/OneClassClassification_20181010/DeepAutoEncoder_Pytorch/
### ------------------------------------------------------------------------------------------
### Case3
##python3 main_autoencoder_case3_train_set_without_abnormal_data.py -i "{'normal_files': ['../input_data/sess_normal_0.txt'], 'attack_files': ['../input_data/sess_TDL4_HTTP_Requests_0.txt', '../input_data/sess_Rcv_Wnd_Size_0_0.txt']}" -e 1000 -o '../log'
#python3 main_autoencoder_case3_train_set_without_abnormal_data.py -i "{'normal_files': ['../input_data/sess_normal_0.txt'], 'attack_files': ['../input_data/sess_DDoS_Excessive_GET_POST']}" -e 1000 -o '../log'
#python3 main_autoencoder_case3_train_set_without_abnormal_data.py -i "{'normal_files': ['../input_data/sess_normal_0.txt'], 'attack_files': ['../input_data/sess_DDoS_Multi_VERB_Single_Request']}" -e 1000 -o '../log'
#python3 main_autoencoder_case3_train_set_without_abnormal_data.py -i "{'normal_files': ['../input_data/sess_normal_0.txt'], 'attack_files': ['../input_data/sess_DDoS_Recursive_GET']}" -e 1000 -o '../log'
#python3 main_autoencoder_case3_train_set_without_abnormal_data.py -i "{'normal_files': ['../input_data/sess_normal_0.txt'], 'attack_files': ['../input_data/sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait']}" -e 1000 -o '../log'
#python3 main_autoencoder_case3_train_set_without_abnormal_data.py -i "{'normal_files': ['../input_data/sess_normal_0.txt'], 'attack_files': ['../input_data/sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait']}" -e 1000 -o '../log'

### ------------------------------------------------------------------------------------------
#### Case4
#python3 main_autoencoder_case4_train_set_without_abnormal_data.py -i "{'normal_files': ['../input_data/sess_normal_0.txt'], \
#'attack_files': ['../input_data/sess_DDoS_Excessive_GET_POST','../input_data/sess_DDoS_Multi_VERB_Single_Request','../input_data/sess_DDoS_Recursive_GET', \
#'../input_data/sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait','../input_data/sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait']}" -e 1000 -o '../log'

### ------------------------------------------------------------------------------------------
#### Case5
python3 main_autoencoder_case5_train_test_only_normal_data.py -i "{'normal_files':  ['../input_data/file1_70M_1.txt'],
'attack_files': ['../input_data/file2_70M_1.txt','../input_data/sess_normal_0.txt','../input_data/sess_DDoS_Excessive_GET_POST',
'../input_data/sess_DDoS_Multi_VERB_Single_Request','../input_data/sess_DDoS_Recursive_GET','../input_data/sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait'
,'../input_data/sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait']}" -e 1000 -o '../log'
