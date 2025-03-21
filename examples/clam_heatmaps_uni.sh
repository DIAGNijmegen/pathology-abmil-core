#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --qos=low
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=20G
#SBATCH --time=4:00:00
#SBATCH --container-mounts=/data/pa_cpgarchive:/data/pa_cpgarchive,/data/temporary:/data/temporary
#SBATCH --container-image="dodrio1.umcn.nl#uokbaseimage/base"
#SBATCH --output=/home/ivanslootweg/logs/clam-heatmaps%j.out
#SBATCH --requeue
#SBATCH --exclude=dlc-mewtwo,dlc-meowth

bash ./environment/create_environment.sh

CUDA_VISIBLE_DEVICES=0,1 

python3 "./create_heatmaps_cobra.py" \
    --save_exp_code="bcc_bin_uni_1e5_s1" \
    --overlap 0 \
    --config_file="/data/temporary/ivan/cloned_tools/CLAM/heatmaps/configs/config_template_cobra.yaml" \