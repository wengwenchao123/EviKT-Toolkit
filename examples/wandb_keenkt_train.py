import argparse

from wandb_train import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 基础训练配置
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--model_name", type=str, default="keenkt")
    parser.add_argument("--emb_type", type=str, default="stoc_qid")
    parser.add_argument("--use_question_difficulty", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save_dir", type=str, default="saved_model_keenkt_edl")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=200)

    # KeenKT网络结构
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--final_fc_dim", type=int, default=256)
    parser.add_argument("--final_fc_dim2", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--kq_same", type=int, choices=[0, 1], default=1)
    parser.add_argument("--atten_type", type=str, default="w2")
    # KeenKT对比学习
    # 专用 KeenKTDataset 提供 r_aug 和 shft_r_aug 增强字段。
    # use_CL 控制对比学习；use_uncertainty_aug 控制是否使用增强标签。
    parser.add_argument("--use_CL", type=int, choices=[0, 1], default=1)
    parser.add_argument("--cl_weight", type=float, default=0.02)
    parser.add_argument("--use_uncertainty_aug", type=int, choices=[0, 1], default=1)

    # KeenKT扩散模块
    # use_diffusion 控制是否启用扩散，diffusion_weight 和 noise_level 控制权重与噪声。
    parser.add_argument("--use_diffusion", type=int, choices=[0, 1], default=1)
    parser.add_argument("--diffusion_weight", type=float, default=0.08)
    parser.add_argument("--noise_level", type=float, default=0.3)

    # KeenKT+EDL
    parser.add_argument("--use_edl", type=int, choices=[0, 1], default=1)
    parser.add_argument("--edl_annealing_step", type=int, default=10)
    

    # EDL 预测损失类型：mse 或 bce。
    parser.add_argument("--edl_loss_type", type=str, choices=["mse", "bce"], default="mse")

    parser.add_argument("--num_classes", type=int, choices=[2], default=2)

    # pyKT/WandB
    parser.add_argument("--use_wandb", type=int, choices=[0, 1], default=0)
    parser.add_argument("--add_uuid", type=int, choices=[0, 1], default=0)

    args = parser.parse_args()
    params = vars(args)

    main(params)