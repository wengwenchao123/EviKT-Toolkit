"""
EDL 共用工具，适用于 DKT、AKT、AT-DKT、DKVMN、UKT、KeenKT、simpleKT 和 RobustKT。

当前 8 个模型按二分类使用，类别顺序固定为：
    0 -> 答错 wrong
    1 -> 答对 correct

本文件负责：
    Softplus：logits -> evidence
    Dirichlet 参数：alpha = evidence + 1
    类别概率和不确定性：p = alpha / S，u = K / S
    EDL 期望 MSE 或期望 BCE 预测损失，以及退火加权的 KL 正则项

序列对齐和 smask 筛选由各模型及训练、评估入口处理。
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


def softplus_evidence(logits: torch.Tensor) -> torch.Tensor:
    """
    使用 Softplus 将 logits 转换为非负证据：evidence = softplus(logits)。

    Args:
        logits:
            网络预测层的原始输出，转换后保持形状不变。
            本项目常见形状为 [N, 2]、[B, T, 2] 或 [B, T, num_c, 2]。

    Returns:
        evidence:
            非负证据，形状与 logits 相同。
    """
    return F.softplus(logits)


def dirichlet_from_logits(
    logits: torch.Tensor,
    num_classes: int = 2,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    根据logits计算EDL中的Dirichlet相关结果。

    Args:
        logits:
            网络预测层的原始输出。
            最后一维必须等于类别数。

        num_classes:
            类别数量。
            默认 K=2；本项目的 8 个模型按二分类使用。

    Returns:
        evidence:
            非负类别证据。

        alpha:
            Dirichlet参数，alpha = evidence + 1。

        probabilities:
            类别概率，probabilities = alpha / strength。

        uncertainty:
            总不确定性，uncertainty = K / strength。
            保留最后一个维度，形状为 logits.shape[:-1] + (1,)。
    """
    if logits.size(-1) != num_classes:
        raise ValueError(
            "logits最后一维应该为{}，实际为{}。".format(
                num_classes,
                logits.size(-1),
            )
        )

    # 1. 通过 Softplus 将 logits 转换为非负证据
    evidence = softplus_evidence(logits)

    # 2. 构造Dirichlet参数
    alpha = evidence + 1.0

    # 3. Dirichlet总强度
    strength = alpha.sum(
        dim=-1,
        keepdim=True,
    )

    # 4. 各类别的 Dirichlet 期望概率
    probabilities = alpha / strength

    # 5. 总不确定性
    # 二分类时K=2，因此u=2/S
    uncertainty = float(num_classes) / strength

    return evidence, alpha, probabilities, uncertainty


def kl_divergence(
    alpha: torch.Tensor,
    num_classes: int = 2,
) -> torch.Tensor:
    """
    计算：

        KL(Dir(alpha) || Dir(1))

    Dir(1)表示没有类别偏好的均匀Dirichlet分布。

    Args:
        alpha:
            Dirichlet参数，形状为[N, K]。

        num_classes:
            类别数量K。

    Returns:
        kl:
            每个样本对应一个KL值，形状为[N, 1]。
    """
    if alpha.ndim != 2:
        raise ValueError(
            "alpha必须是[N, K]二维张量，实际形状为{}。".format(
                tuple(alpha.shape)
            )
        )

    if alpha.size(-1) != num_classes:
        raise ValueError(
            "alpha最后一维应该为{}，实际为{}。".format(
                num_classes,
                alpha.size(-1),
            )
        )

    # 均匀Dirichlet分布的参数，全为1
    ones = torch.ones(
        (1, num_classes),
        dtype=alpha.dtype,
        device=alpha.device,
    )

    sum_alpha = alpha.sum(
        dim=-1,
        keepdim=True,
    )

    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(
            dim=-1,
            keepdim=True,
        )
        + torch.lgamma(ones).sum(
            dim=-1,
            keepdim=True,
        )
        - torch.lgamma(
            ones.sum(
                dim=-1,
                keepdim=True,
            )
        )
    )

    second_term = (
        (alpha - ones)
        * (
            torch.digamma(alpha)
            - torch.digamma(sum_alpha)
        )
    ).sum(
        dim=-1,
        keepdim=True,
    )

    return first_term + second_term


def edl_mse_loss(
    logits: torch.Tensor,
    target_one_hot: torch.Tensor,
    epoch_num: int,
    annealing_step: int = 10,
    num_classes: int = 2,
    return_components: bool = False,
    loss_type: str = "mse",
):
    """
    计算 EDL 预测损失与退火加权的 KL 正则项。

    函数名沿用 edl_mse_loss，实际预测损失由 loss_type 决定：
        mse：期望平方误差，即 error + variance。
        bce：Dirichlet 分布下的期望 BCE，即
             sum_k y_k * (digamma(S) - digamma(alpha_k))。

    对当前 batch 的有效位置取平均：
        total_loss = mean(prediction_loss + annealing_coef * KL)

    Args:
        logits:
            模型预测层输出的 logits，形状为 [N, K]。
            本项目 K=2；N 为按 smask 筛选并展平后的有效位置数。
        target_one_hot:
            one-hot 标签，形状与 logits 相同，即 [N, K]。
            本项目中标签 0 答错对应 [1, 0]，标签 1 答对对应 [0, 1]。
        epoch_num:
            从 0 开始的训练轮次索引。
            第一轮传 0，第二轮传 1，第十一轮传 10。
        annealing_step:
            KL 系数达到 1 时的轮次索引，默认值为 10。
            系数为 min(1, epoch_num / annealing_step)。
        num_classes:
            类别数 K，默认值为 2；本项目的 8 个模型按二分类使用。
        return_components:
            是否同时返回损失和诊断统计。
            error、variance 始终是 MSE 分支对应的诊断量；
            loss_type="bce" 时，它们不作为 BCE 预测损失的组成项。
        loss_type:
            "mse" 使用期望 MSE；"bce" 使用期望 BCE。
            两种模式均加入相同构造方式的 KL 正则项。

    Returns:
        return_components=False 时返回 total_loss。
        return_components=True 时返回 (total_loss, components)。
    """
    if logits.ndim != 2:
        raise ValueError(
            "logits必须是[N, K]二维张量，实际形状为{}。".format(
                tuple(logits.shape)
            )
        )

    if logits.size(-1) != num_classes:
        raise ValueError(
            "logits最后一维应该为{}，实际为{}。".format(
                num_classes,
                logits.size(-1),
            )
        )

    if target_one_hot.shape != logits.shape:
        raise ValueError(
            "target_one_hot必须与logits形状相同，实际为{}和{}。".format(
                tuple(target_one_hot.shape),
                tuple(logits.shape),
            )
        )

    if logits.size(0) == 0:
        raise ValueError(
            "smask筛选后没有有效知识点，无法计算EDL损失。"
        )

    target_one_hot = target_one_hot.to(
        device=logits.device,
        dtype=logits.dtype,
    )

    # ==================================================
    # 1. logits -> evidence -> alpha -> probabilities
    # ==================================================

    evidence, alpha, probabilities, uncertainty = (
        dirichlet_from_logits(
            logits=logits,
            num_classes=num_classes,
        )
    )

    strength = alpha.sum(
        dim=-1,
        keepdim=True,
    )

    # ==================================================
    # 2. 概率误差项
    #
    # sum_k (y_k - alpha_k / S)^2
    # ==================================================

    error = (
        (target_one_hot - probabilities) ** 2
    ).sum(
        dim=-1,
        keepdim=True,
    )

    # ==================================================
    # 3. Dirichlet方差项
    #
    # sum_k alpha_k(S-alpha_k) / [S^2(S+1)]
    # ==================================================

    variance = (
        alpha
        * (strength - alpha)
        / (
            strength
            * strength
            * (strength + 1.0)
        )
    ).sum(
        dim=-1,
        keepdim=True,
    )

    # ==================================================
    # 4. 构造KL使用的alpha
    #
    # 真实类别的alpha被替换为1；
    # 错误类别的alpha保留。
    #
    # 这样KL主要约束错误类别上的多余证据。
    # ==================================================

    kl_alpha = (
        (alpha - 1.0)
        * (1.0 - target_one_hot)
        + 1.0
    )

    kl = kl_divergence(
        alpha=kl_alpha,
        num_classes=num_classes,
    )

    # ==================================================
    # 5. KL退火系数
    #
    # epoch_num从0开始：
    #
    # epoch 0  -> 0.0
    # epoch 1  -> 0.1
    # epoch 5  -> 0.5
    # epoch 10 -> 1.0
    # ==================================================

    annealing_step = max(
        int(annealing_step),
        1,
    )

    epoch_num = max(
        int(epoch_num),
        0,
    )

    annealing_coef = min(
        1.0,
        float(epoch_num)
        / float(annealing_step),
    )

    # ==================================================
    # 6. 完整EDL损失
    # ==================================================

    # 按 loss_type 选择预测损失，再加入退火加权的 KL 正则项。
    loss_type = str(loss_type).lower()

    if loss_type == "mse":
        # 原来的期望 MSE：平方误差 + Dirichlet 方差
        prediction_loss = error + variance

    elif loss_type == "bce":
        # Dirichlet 分布下的期望 BCE
        prediction_loss = (
            target_one_hot
            * (
                torch.digamma(strength)
                - torch.digamma(alpha)
            )
        ).sum(dim=-1, keepdim=True)

    else:
        raise ValueError(
            f"Unsupported EDL loss_type: {loss_type}. "
            "Choose 'mse' or 'bce'."
        )

    total_per_sample = (
        prediction_loss
        + annealing_coef * kl
    )


    # 对当前batch全部有效知识点位置取平均
    total_loss = total_per_sample.mean()

    if not return_components:
        return total_loss

    # detach 只用于保存和打印诊断统计，不影响 total_loss 的反向传播。
    # error、variance 在 BCE 模式下仍记录 MSE 诊断量，不用于重构 BCE 总损失。
    components: Dict[str, Any] = {
        "valid_count": int(
            logits.size(0)
        ),
        "total": total_loss.detach(),
        "error": error.mean().detach(),
        "variance": variance.mean().detach(),
        "kl": kl.mean().detach(),
        "annealing_coef": annealing_coef,
        "probability_min": probabilities.min().detach(),
        "probability_max": probabilities.max().detach(),
        "probability_sum_min": (
            probabilities.sum(dim=-1).min().detach()
        ),
        "probability_sum_max": (
            probabilities.sum(dim=-1).max().detach()
        ),
        "uncertainty_mean": uncertainty.mean().detach(),
        "strength_mean": strength.mean().detach(),
        "evidence_mean": evidence.mean().detach(),

        "zero_evidence_ratio": (evidence.sum(dim=-1)== 0.0).float().mean().detach()

    }

    return total_loss, components

