
python3 "/data/temporary/ivan/cloned_tools/CLAM/run_ood_detection.py" \
    --data_root_dir="/data/temporary/natalia/wk_3/UNI" \
    --results_dir="/data/temporary/ivan/DeepDerma/BCC_CLAM" \
    --data_label_csv_path_test="/data/temporary/ivan/DeepDerma/OOD/splits/ood_dataset_val.csv" \
    --data_label_csv_path_train="/data/temporary/ivan/DeepDerma/OOD/splits/ood_dataset_train.csv" \
    --exp_code="bcc_bin_uni_1e5" \
    --tasks "cobra_vs_otherdiseases" "cobra_vs_scc" \
    --split "val" \
    --seed 1 \
    --max_epochs 9 \
    --lr 1e-5 \
    --reg 1e-5 \
    --label_frac 1.0 \
    --drop_out 0.25 \
    --bag_loss ce \
    --inst_loss svm \
    --opt adam \
    --weighted_sample \
    --model_type addmil \
    --model_size small \
    --bag_weight 0.7 \
    --B 8 \
    --embed_dim 1024 \
    --datatype "npy" 
    
# for testing vs various id/ood sets, the only thing that has to be change is the tassk parameter. The other values can remain. --tasks takes care of subsetting the ood_dataset.csv