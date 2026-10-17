# pyKT with EDL

This project retains DKT, AKT, AT-DKT, DKVMN, UKT, KeenKT, simpleKT and
RobustKT, together with their shared data preprocessing and EDL tools.

EDL uses Softplus evidence. Test evaluation reports AUC, ACC, MUD and W2.
MUD is the mean uncertainty of incorrect predictions minus that of correct
predictions. W2 compares the two empirical uncertainty distributions.
There are no EDL plots, selective prediction experiments or per-sample EDL CSVs.

Run commands from the `examples` directory after configuring dataset paths in
`configs/data_config.json`.

```sh
python wandb_dkt_train.py --dataset_name assist2015 --use_edl 1 --use_wandb 0
python wandb_predict.py --save_dir <trained_model_directory> --use_wandb 0
```

Replace `dkt` in the training filename with one of the supported model names.
UKT and KeenKT use the `stoc_qid` embedding type by default.
Training uses only training and validation data and saves `config.json` and
`<emb_type>_model.ckpt`. Testing uses the saved model and reports ordinary and
windowed test metrics.

Testing writes only these metric tables:

```text
<trained_model_directory>/
  <emb_type>_final_test_metrics.csv
  edl_analysis/test_metrics.csv
  edl_analysis/test_window_metrics.csv
```

The two individual metric tables have columns `auc,acc,mud,w2`. The final
table contains both test settings plus saved configuration metadata.
MUD and W2 are undefined when all predictions are correct or all are incorrect;
AUC is undefined when only one true label occurs. Such values are reported as NaN.

Dataset preprocessing modules are retained independently of the model list.
