# EDL 模型训练与测试

保留模型：DKT、AKT、AT-DKT、DKVMN、UKT、KeenKT、simpleKT、RobustKT。
在 `configs/data_config.json` 配置数据路径，进入 `examples` 目录运行：

```sh
python wandb_dkt_train.py --dataset_name assist2015 --use_edl 1 --use_wandb 0
python wandb_predict.py --save_dir <模型保存目录> --use_wandb 0
```

将训练脚本名中的 `dkt` 换成对应模型名。UKT、KeenKT 默认使用 `stoc_qid`。
训练使用训练集和验证集，保存配置及最佳权重；测试加载已保存的模型。
测试指标只有 AUC、ACC、MUD、W2，普通测试和窗口测试分别计算。
MUD 为预测错误样本平均不确定性减去预测正确样本平均不确定性。
W2 为这两组不确定性经验分布之间的二阶 Wasserstein 距离。

测试只生成两张四指标表和最终汇总表，不输出逐样本 EDL CSV、图片或选择性预测结果。
共用的数据预处理代码仍然保留。模型完整备份位于项目外的 `D:/pykt+edl/code-backups`。
