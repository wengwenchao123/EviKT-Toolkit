import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--model_name", type=str, default="robustkt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--ks", type=int, default=5)

    # RobustKT+EDL配置
    # 是否启用 EDL：1 开启，0 使用原始模型。
    parser.add_argument("--use_edl", type=int, choices=[0, 1], default=1)

    # EDL 的 KL 系数增至 1 所需的轮数。
    parser.add_argument("--edl_annealing_step", type=int, default=10)

    # 二分类：0 表示答错，1 表示答对。
    parser.add_argument("--num_classes", type=int, choices=[2], default=2)


    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--add_uuid", type=int, default=0)
   
    args = parser.parse_args()

    params = vars(args)
    main(params)
