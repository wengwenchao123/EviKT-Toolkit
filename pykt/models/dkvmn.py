import torch

from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_

from .edl import dirichlet_from_logits


class DKVMN(Module):
    def __init__(
        self,
        num_c,
        dim_s,
        size_m,
        dropout=0.2,
        emb_type="qid",
        emb_path="",
        pretrain_dim=768,
        use_edl=False,
        edl_annealing_step=10,
        num_classes=2,
    ):
        super().__init__()

        self.model_name = "dkvmn"
        self.num_c = num_c
        self.dim_s = dim_s
        self.size_m = size_m
        self.emb_type = emb_type

        # EDL配置
        self.use_edl = bool(use_edl)
        self.edl_annealing_step = edl_annealing_step
        self.num_classes = num_classes

        if emb_type.startswith("qid"):
            # 知识点嵌入
            self.k_emb_layer = Embedding(
                self.num_c,
                self.dim_s,
            )

            # 静态Key Memory
            self.Mk = Parameter(
                torch.Tensor(
                    self.size_m,
                    self.dim_s,
                )
            )

            # 动态Value Memory的初始状态
            self.Mv0 = Parameter(
                torch.Tensor(
                    self.size_m,
                    self.dim_s,
                )
            )

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        # 知识点和作答结果的交互嵌入
        self.v_emb_layer = Embedding(
            self.num_c * 2,
            self.dim_s,
        )

        self.f_layer = Linear(
            self.dim_s * 2,
            self.dim_s,
        )

        self.dropout_layer = Dropout(dropout)

        # 原始DKVMN输出一个logit
        # EDL输出答错、答对两个logit
        if self.use_edl:
            self.p_layer = Linear(
                self.dim_s,
                self.num_classes,
            )
        else:
            self.p_layer = Linear(
                self.dim_s,
                1,
            )

        self.e_layer = Linear(
            self.dim_s,
            self.dim_s,
        )

        self.a_layer = Linear(
            self.dim_s,
            self.dim_s,
        )

    def forward(
        self,
        q,
        r,
        qtest=False,
        return_edl=False,
    ):
        """
        参数
        ----------
        q:
            知识点序列，形状 [batch_size, seq_len]

        r:
            作答结果序列，形状 [batch_size, seq_len]

        qtest:
            是否返回题目级融合需要的隐藏表示f

        return_edl:
            是否额外返回EDL相关结果

        返回
        ----------
        普通DKVMN:
            p，形状 [batch_size, seq_len]

        DKVMN+EDL:
            p为答对类别概率，形状 [batch_size, seq_len]

            qtest=False、return_edl=False 时返回 p。
            qtest=False、return_edl=True 时返回 (p, edl_output)。
            qtest=True 时额外返回隐藏表示 f：
                return_edl=False：返回 (p, f)。
                return_edl=True：返回 (p, f, edl_output)。
            edl_output 为字典，包含：
                logits、evidence、alpha、probabilities：[B,T,2]
                uncertainty、strength：[B,T]
            p 为答对概率；字典中的 probabilities 为两类概率。
        """

        batch_size = q.shape[0]

        if self.emb_type.startswith("qid"):
            # r=0时使用前num_c个嵌入
            # r=1时使用后num_c个嵌入
            x = q + self.num_c * r

            k = self.k_emb_layer(q)
            v = self.v_emb_layer(x)
        else:
            raise ValueError(
                f"DKVMN does not support emb_type={self.emb_type}"
            )

        # 初始化每个学生的Value Memory
        Mvt = self.Mv0.unsqueeze(0).repeat(
            batch_size,
            1,
            1,
        )

        Mv = [Mvt]

        # 知识点对每个Memory Slot的相关性权重
        w = torch.softmax(
            torch.matmul(k, self.Mk.T),
            dim=-1,
        )

        # Write Process
        erase = torch.sigmoid(
            self.e_layer(v)
        )

        add = torch.tanh(
            self.a_layer(v)
        )

        for erase_t, add_t, weight_t in zip(
            erase.permute(1, 0, 2),
            add.permute(1, 0, 2),
            w.permute(1, 0, 2),
        ):
            Mvt = (
                Mvt
                * (
                    1
                    - weight_t.unsqueeze(-1)
                    * erase_t.unsqueeze(1)
                )
                + weight_t.unsqueeze(-1)
                * add_t.unsqueeze(1)
            )

            Mv.append(Mvt)

        Mv = torch.stack(
            Mv,
            dim=1,
        )

        # Read Process
        read_content = (
            w.unsqueeze(-1)
            * Mv[:, :-1]
        ).sum(-2)

        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        read_content,
                        k,
                    ],
                    dim=-1,
                )
            )
        )

        logits = self.p_layer(
            self.dropout_layer(f)
        )

        # =====================================================
        # DKVMN + EDL
        # =====================================================
        if self.use_edl:
            (
                evidence,
                alpha,
                probabilities,
                uncertainty,
            ) = dirichlet_from_logits(
                logits,
                num_classes=self.num_classes,
            )

            # 类别顺序：
            # 0 -> 答错
            # 1 -> 答对
            p = probabilities[..., 1]

            strength = alpha.sum(
                dim=-1
            )

            edl_output = {
                "logits": logits,
                "evidence": evidence,
                "alpha": alpha,
                "probabilities": probabilities,
                "uncertainty": uncertainty.squeeze(-1),
                "strength": strength,
            }

            # 题目级预测需要隐藏状态f
            if qtest:
                if return_edl:
                    return p, f, edl_output

                return p, f

            # 训练或者知识点级EDL测试
            if return_edl:
                return p, edl_output

            return p

        # =====================================================
        # 原始DKVMN
        # =====================================================
        p = torch.sigmoid(logits)
        p = p.squeeze(-1)

        if return_edl:
            raise ValueError(
                "return_edl=True requires use_edl=True"
            )

        if qtest:
            return p, f

        return p