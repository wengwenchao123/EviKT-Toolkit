from .data_loader import KTDataset


class KeenKTDataset(KTDataset):
    """
    KeenKT专用数据集。

    保留pyKT原有KTDataset的数据读取和五折划分，
    只增加原版KeenKT对比学习所需的：
        r_aug
        shft_r_aug
    """

    def __getitem__(self, index):
        sample = super().__getitem__(index)

        # 兼容知识点级和题目级数据返回形式
        if isinstance(sample, tuple):
            dcur, dqtest = sample
        else:
            dcur = sample
            dqtest = None

        masks = dcur["masks"].bool()

        r_aug = dcur["rseqs"].clone()
        rshft_aug = dcur[
            "shft_rseqs"
        ].clone()

        valid_r = r_aug[masks]
        valid_rshft = rshft_aug[masks]

        # 严格复现原版KeenKT的数据增强逻辑
        if (
            valid_r.numel() > 0
            and valid_rshft.numel() > 0
        ):
            if valid_rshft[-1] == 1:
                if valid_r[0] == 1:
                    r_aug[0] = 1 - r_aug[0]

                for i in range(
                    len(valid_rshft)
                ):
                    if rshft_aug[0] == 1:
                        rshft_aug[i] = (
                            1 - rshft_aug[i]
                        )

            elif valid_rshft[-1] == 0:
                if valid_r[0] == 0:
                    r_aug[0] = 1 - r_aug[0]

                for i in range(
                    len(valid_rshft)
                ):
                    if rshft_aug[0] == 0:
                        rshft_aug[i] = (
                            1 - rshft_aug[i]
                        )

        dcur["r_aug"] = r_aug
        dcur["shft_r_aug"] = rshft_aug

        if dqtest is not None:
            return dcur, dqtest

        return dcur