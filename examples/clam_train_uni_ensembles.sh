#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --time=1-00:00:00
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/temporary:/data/temporary
#SBATCH --container-image="dodrio1.umcn.nl#uokbaseimage/base" 
#SBATCH --output=/home/ivanslootweg/logs/clam-%j.out
#SBATCH --requeue
#SBATCH --exclude=dlc-mewtwo,dlc-meowth
#SBATCH --qos=high

cd /data/temporary/ivan/cpgscriptbackup/ood_detection/
bash /data/temporary/ivan/cpgscriptbackup/ood_detection/ood_detection_env.sh

cd /data/temporary/ivan/cpgscriptbackup/clam/
CUDA_VISIBLE_DEVICES=0,1 

for seed in {0..4}; do
    k_end=$((seed + 1))
    python3 "/data/temporary/ivan/cloned_tools/CLAM/main.py" \
        --data_root_dir="/data/temporary/natalia/wk_3/UNI/cobra_UNI" \
        --results_dir="/data/temporary/ivan/DeepDerma/BCC_CLAM" \
        --split_dir="/data/temporary/ivan/DeepDerma/classifier_splits/cobra_kfold/0/" \
        --data_label_csv_path="/data/temporary/ivan/DeepDerma/classifier_splits/cobra_clam_format/bcc_bin.csv" \
        --exp_code="bcc_bin_uni_1e5_kfold" \
        --task bcc_bin \
        --seed "$seed" \
        --max_epochs 40 \
        --lr 1e-5 \
        --reg 1e-5 \
        --label_frac 1.0 \
        --k_start $seed \
        --k_end $k_end \
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

done
        
