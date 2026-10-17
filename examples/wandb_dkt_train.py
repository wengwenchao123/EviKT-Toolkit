import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_name", type=str, default="dkt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--emb_size", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    # DKT+EDL配置
    # 是否启用 EDL：1 开启，0 使用原始模型。
    parser.add_argument("--use_edl", type=int, default=0)

    # EDL 的 KL 系数增至 1 所需的轮数。
    parser.add_argument("--edl_annealing_step", type=int, default=10)
    # EDL 预测损失类型：mse 或 bce。
    parser.add_argument("--edl_loss_type", type=str, choices=["mse", "bce"], default="mse")


    # 二分类：0 表示答错，1 表示答对。
    parser.add_argument("--num_classes", type=int, default=2)


    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    
    args = parser.parse_args()

    params = vars(args)
    main(params)
