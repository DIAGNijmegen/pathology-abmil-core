#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --qos=low
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=20G
#SBATCH --time=10:00:00
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/temporary:/data/temporary
#SBATCH --container-image="doduo.umcn.nl#daangeijs/cobra/cobrastreamingclam:latest"
#SBATCH --output=/home/ivanslootweg/logs/clam-%j.out
#SBATCH --requeue
#SBATCH --exclude=dlc-mewtwo,dlc-meowth

cd /data/temporary/ivan/cpgscriptbackup/clam/
bash /data/temporary/ivan/cpgscriptbackup/clam/clam_env.sh

CUDA_VISIBLE_DEVICES=0,1 

python3 "/data/temporary/ivan/cloned_tools/CLAM/main.py" \
    --data_root_dir="/data/temporary/natalia/wk_3/UNI/cobra_UNI" \
    --results_dir="/data/temporary/ivan/DeepDerma/BCC_CLAM" \
    --split_dir="/data/temporary/ivan/DeepDerma/BCC_CLAM/data"\
    --data_label_csv_path="/data/temporary/ivan/DeepDerma/BCC_CLAM/data/bcc.csv" \
    --exp_code="bcc_bin_uni_1e5_v1" \
    --task bcc_bin \
    --seed 1 \
    --max_epochs 9 \
    --lr 1e-5 \
    --reg 1e-5 \
    --label_frac 1.0 \
    --k 1 \
    --k_start 0 \
    --drop_out 0.25 \
    --bag_loss ce \
    --inst_loss svm \
    --opt adam \
    --weighted_sample \
    --model_type addmil \
    --model_size small \
    --bag_weight 0.7 \
    --B 8 \
    --log_data \
    --use_wandb \
    --embed_dim 1024 \
    --datatype "npy" \
    --resume \
    --early_stopping 
    