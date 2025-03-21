
python3 "./tsne_projections.py" \
    --data_root_dir="/data/temporary/natalia/wk_3/UNI" \
    --results_dir="/data/temporary/ivan/DeepDerma/BCC_CLAM" \
    --data_label_csv_path="/data/temporary/ivan/DeepDerma/OOD/splits/ood_dataset_val.csv" \
    --exp_code="bcc_bin_uni_1e5" \
    --tasks cobra_vs_otherdiseases \
    --seed 1 \
    --max_epochs 9 \
    --split "trainval" \
    --drop_out 0.25 \
    --bag_loss ce \
    --inst_loss svm \
    --weighted_sample \
    --model_type addmil \
    --model_size small \
    --bag_weight 0.7 \
    --B 8 \
    --embed_dim 1024 \
    --datatype "npy" \