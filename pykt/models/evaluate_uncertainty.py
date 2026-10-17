import os


import numpy as np
import pandas as pd

from sklearn import metrics


# analyze_edl_uncertainty真正需要的原始字段。
# pred_label和is_error会在分析时重新计算。
REQUIRED_COLUMNS = {
    "true_label",
    "pred_probability",
    "uncertainty",
}


def _validate_result_df(result_df):
    """
    检查evaluate_edl()输出是否合法。
    """
    if not isinstance(
        result_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "result_df must be a pandas DataFrame."
        )

    if result_df.empty:
        raise ValueError(
            "EDL result DataFrame is empty."
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(result_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "EDL result is missing columns: "
            f"{sorted(missing_columns)}"
        )

    true_labels = (
        result_df["true_label"]
        .astype(np.int64)
        .to_numpy()
    )

    if not np.isin(
        true_labels,
        [0, 1],
    ).all():
        raise ValueError(
            "true_label must contain only 0 and 1."
        )

    for column in [
        "pred_probability",
        "uncertainty",
    ]:
        values = (
            result_df[column]
            .astype(float)
            .to_numpy()
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{column} contains NaN or Inf."
            )


def calculate_empirical_w2(
    first_values,
    second_values,
):
    """
    精确计算两个一维经验分布之间的W2。

    W2 = sqrt(
        integral_0^1
        (Q_first(p) - Q_second(p))^2 dp
    )

    这里计算的是W2，不是scipy中的
    wasserstein_distance，因为后者是W1。
    """
    first = np.sort(
        np.asarray(
            first_values,
            dtype=np.float64,
        )
    )

    second = np.sort(
        np.asarray(
            second_values,
            dtype=np.float64,
        )
    )

    first_count = int(
        first.size
    )

    second_count = int(
        second.size
    )

    if (
        first_count == 0
        or second_count == 0
    ):
        return float("nan")

    first_index = 0
    second_index = 0
    previous_probability = 0.0
    squared_integral = 0.0

    while (
        first_index < first_count
        and second_index < second_count
    ):
        # 使用整数交叉乘法比较两个
        # 经验分位数函数的下一个断点。
        first_boundary = (
            (first_index + 1)
            * second_count
        )

        second_boundary = (
            (second_index + 1)
            * first_count
        )

        if (
            first_boundary
            < second_boundary
        ):
            next_probability = (
                (first_index + 1)
                / first_count
            )

            advance_first = True
            advance_second = False

        elif (
            second_boundary
            < first_boundary
        ):
            next_probability = (
                (second_index + 1)
                / second_count
            )

            advance_first = False
            advance_second = True

        else:
            next_probability = (
                (first_index + 1)
                / first_count
            )

            advance_first = True
            advance_second = True

        difference = (
            first[first_index]
            - second[second_index]
        )

        squared_integral += (
            next_probability
            - previous_probability
        ) * difference ** 2

        previous_probability = (
            next_probability
        )

        if advance_first:
            first_index += 1

        if advance_second:
            second_index += 1

    return float(
        np.sqrt(
            max(
                squared_integral,
                0.0,
            )
        )
    )


def calculate_mean_uncertainty(
    result_df,
):
    """
    比较预测正确与预测错误样本的平均uncertainty。
    """
    correct_uncertainty = result_df.loc[
        result_df["is_error"] == 0,
        "uncertainty",
    ].astype(float)

    wrong_uncertainty = result_df.loc[
        result_df["is_error"] == 1,
        "uncertainty",
    ].astype(float)

    mean_u_correct = (
        float(correct_uncertainty.mean())
        if len(correct_uncertainty) > 0
        else float("nan")
    )

    mean_u_wrong = (
        float(wrong_uncertainty.mean())
        if len(wrong_uncertainty) > 0
        else float("nan")
    )

    mean_u_gap = (
        mean_u_wrong
        - mean_u_correct
    )

    w2_distance = (
        calculate_empirical_w2(
            correct_uncertainty.to_numpy(
                dtype=np.float64
            ),
            wrong_uncertainty.to_numpy(
                dtype=np.float64
            ),
        )
    )


    return {
        "correct_count": int(
            len(correct_uncertainty)
        ),
        "wrong_count": int(
            len(wrong_uncertainty)
        ),
        "mean_u_correct": (
            mean_u_correct
        ),
        "mean_u_wrong": (
            mean_u_wrong
        ),
        "mean_u_gap": (
            mean_u_gap
        ),

        "w2": w2_distance,
    }


def analyze_edl_uncertainty(result_df, output_dir="", prefix="test"):
    """Report only AUC, ACC, MUD and W2, without filtering predictions.

    MUD is mean uncertainty of incorrect predictions minus that of correct
    predictions. MUD and W2 are NaN when either group is empty.
    """
    _validate_result_df(result_df)
    result_df = result_df.copy()
    targets = result_df["true_label"].astype(np.int64).to_numpy()
    probabilities = result_df["pred_probability"].astype(float).to_numpy()
    predictions = (probabilities >= 0.5).astype(np.int64)
    result_df["is_error"] = (predictions != targets).astype(np.int64)
    mean_result = calculate_mean_uncertainty(result_df)
    summary = {
        "auc": float(metrics.roc_auc_score(targets, probabilities))
        if np.unique(targets).size >= 2 else float("nan"),
        "acc": float(metrics.accuracy_score(targets, predictions)),
        "mud": mean_result["mean_u_gap"],
        "w2": mean_result["w2"],
    }

    summary_path = ""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, prefix + "_metrics.csv")
        pd.DataFrame([summary]).to_csv(
            summary_path, index=False, encoding="utf-8-sig"
        )

    print(
        f"{prefix} metrics | AUC={summary['auc']:.6f}, "
        f"ACC={summary['acc']:.6f}, MUD={summary['mud']:.6f}, "
        f"W2={summary['w2']:.6f}"
    )
    return {"summary": summary, "summary_path": summary_path}
