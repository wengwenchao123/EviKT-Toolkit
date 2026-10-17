import torch

from torch.nn import (
    Module,
    Embedding,
    LSTM,
    Linear,
    Dropout,
)

from .edl import (
    dirichlet_from_logits,
)


class DKT(Module):
    def __init__(
        self,
        num_c,
        emb_size,
        dropout=0.1,
        emb_type="qid",
        emb_path="",
        pretrain_dim=768,
        use_edl=False,
        edl_annealing_step=10,
        num_classes=2,
        edl_loss_type="mse",
    ):
        super().__init__()

        self.model_name = "dkt"
        self.num_c = int(num_c)
        self.emb_size = int(emb_size)
        self.hidden_size = int(
            emb_size
        )
        self.emb_type = emb_type

        # =================================================
        # EDL配置
        # =================================================
        self.use_edl = bool(
            use_edl
        )

        self.edl_annealing_step = int(
            edl_annealing_step
        )
        self.edl_loss_type = str(edl_loss_type).lower()

        if self.edl_loss_type not in ("mse", "bce"):
            raise ValueError(
                "edl_loss_type must be 'mse' or 'bce'."
            )

        if self.use_edl:
            print(
                f"EDL prediction loss: {self.edl_loss_type}, "
                f"KL annealing step: {self.edl_annealing_step}"
            )


        self.num_classes = int(
            num_classes
        )

        if (
            self.use_edl
            and self.num_classes != 2
        ):
            raise ValueError(
                "DKT+EDL currently supports "
                "binary classification only. "
                "num_classes must be 2."
            )

        # =================================================
        # DKT交互嵌入
        # interaction_id = concept + num_c * response
        # =================================================
        if self.emb_type.startswith(
            "qid"
        ):
            self.interaction_emb = (
                Embedding(
                    self.num_c * 2,
                    self.emb_size,
                )
            )

        self.lstm_layer = LSTM(
            self.emb_size,
            self.hidden_size,
            batch_first=True,
        )

        self.dropout_layer = Dropout(
            dropout
        )

        # =================================================
        # 原始DKT：
        #   每个知识点输出一个概率logit
        #   输出维度为num_c
        #
        # DKT+EDL：
        #   每个知识点输出两个类别logits
        #   输出维度为num_c * 2
        # =================================================
        output_dim = (
            self.num_c
            * self.num_classes
            if self.use_edl
            else self.num_c
        )

        self.out_layer = Linear(
            self.hidden_size,
            output_dim,
        )

    def forward(
        self,
        q,
        r,
        return_edl=False,
    ):
        """
        Args:
            q:
                历史知识点序列，[B,T]。

            r:
                历史作答序列，[B,T]。

            return_edl:
                是否额外返回EDL结果。

        Returns:
            use_edl=False:
                probabilities，[B,T,num_c]。

            use_edl=True:
                probabilities，[B,T,num_c]。

                return_edl=False 时只返回 probabilities。
                return_edl=True 时返回 (probabilities, edl_output)。
                edl_output 为字典，包含：
                    logits、evidence、alpha、probabilities：[B,T,num_c,2]
                    strength、uncertainty：[B,T,num_c]
                外层 probabilities 为答对概率；字典中的 probabilities 为两类概率。
        """
        if self.emb_type.startswith(
            "qid"
        ):
            interaction_ids = (
                q
                + self.num_c * r
            )

            xemb = self.interaction_emb(
                interaction_ids
            )

        else:
            raise ValueError(
                "DKT currently supports "
                "qid embedding only, "
                f"got {self.emb_type}."
            )

        hidden, _ = self.lstm_layer(
            xemb
        )

        hidden = self.dropout_layer(
            hidden
        )

        raw_output = self.out_layer(
            hidden
        )

        # =================================================
        # DKT+EDL
        # =================================================
        if self.use_edl:
            batch_size = (
                raw_output.size(0)
            )

            sequence_length = (
                raw_output.size(1)
            )

            # [B,T,num_c*2]
            #          ↓
            # [B,T,num_c,2]
            logits = raw_output.reshape(
                batch_size,
                sequence_length,
                self.num_c,
                self.num_classes,
            )

            (
                evidence,
                alpha,
                class_probabilities,
                uncertainty,
            ) = dirichlet_from_logits(
                logits=logits,
                num_classes=(
                    self.num_classes
                ),
            )

            # 最后一维类别：
            # 0 = 答错
            # 1 = 答对
            probabilities = (
                class_probabilities[
                    ...,
                    1,
                ]
            )

            edl_output = {
                "logits": logits,
                "evidence": evidence,
                "alpha": alpha,
                "probabilities": (
                    class_probabilities
                ),
                "strength": alpha.sum(
                    dim=-1
                ),
                "uncertainty": (
                    uncertainty.squeeze(-1)
                ),
            }

            if return_edl:
                return (
                    probabilities,
                    edl_output,
                )

            return probabilities

        # =================================================
        # 原始DKT
        # =================================================
        probabilities = torch.sigmoid(
            raw_output
        )

        return probabilities