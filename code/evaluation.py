import sys
import os
from evaluation.settings import benchmark_settings
from evaluation.utils.common import common_args
from evaluation.utils.evaluator_main import evaluator, prepare_results
from evaluation.utils.postprocess import post_average


datasets_full = [
    "Proxifier",
    "Apache",
    "OpenSSH",
    "HDFS",
    "OpenStack",
    "HPC",
    "Zookeeper",
    "HealthApp",
    "Hadoop",
    "Spark",
    "BGL",
    "Linux",
    "Mac",
    "Thunderbird",
]


if __name__ == "__main__":
    args = common_args()
    datasets = datasets_full
    data_type = "full"
    input_dir = "../datasets/loghub-2.0-full/"
    output_dir = f"{args.output_dir}/logs"
    if not os.path.exists(output_dir):
        raise FileNotFoundError(
            f"Output directory {output_dir} does not exist.")

    result_file = prepare_results(output_dir=output_dir)
    if args.dataset != "null":
        datasets = [args.dataset]

    for dataset in datasets:
        setting = benchmark_settings[dataset]
        log_file = setting['log_file'].replace("_2k", f"_{data_type}")
        if os.path.exists(os.path.join(output_dir, f"{dataset}.log_structured.csv")):
            raise FileExistsError(
                f"parsing result of dataset {dataset} not exist.")

        evaluator(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            log_file=log_file,
            result_file=result_file,
        )
    metric_file = os.path.join(output_dir, result_file)

    if args.dataset == "null":
        post_average(metric_file)
