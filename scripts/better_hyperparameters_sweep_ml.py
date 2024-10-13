from dataclasses import dataclass, field
import time
import os
import stat
from datetime import datetime

import wandb
from src.configs.constants import MLModelNames
from run_wrapper import (
    MLModelOptions,
    MLTrainerOptions,
    DataOptions,
    DataPathOptions,
)
from typing import Tuple, TypedDict
import argparse


"""
Usage:

1. check that 'search_space_by_model_name' has the correct hyperparameter search space for the model you wish to sweep.
2. add a new 'RunConfig' to 'run_configs' with the desired hyperparameters.
3. run 'python scripts/better_hyperparameters_sweep.py --config-name <config_name> --run-cap <run_cap> --wandb-project <wandb_project> --wandb-entity <wandb_entity>'
    * notes:
    * * 'config_name' should be the key of the 'RunConfig' in 'run_configs'.
    * * 'config_name' is the only required argument.
3.cont. the script will create an executable bash script for each sweep (fold_idx), which will launch the wandb sweeps.
4. run the bash script ./<bash_script>.sh
5. input the GPU index and the number of runs you want to launch on that GPU to start the sweeps.
"""


class MLRunConfig(TypedDict):
    model_name: MLModelNames
    model_variant: MLModelOptions
    data_variant: DataOptions
    data_path: DataPathOptions
    trainer_variant: MLTrainerOptions
    add_feature_search_space: bool


@dataclass
class HyperArgs:
    # model options
    model_name: MLModelNames  # i.e., model logic
    model_variant: MLModelOptions  # i.e., variant of the model (e.g., PredCfg)
    add_feature_search_space: bool

    # data options
    data_variant: DataOptions
    data_path: str

    # training options
    trainer_variant: MLTrainerOptions

    # wandb options
    wandb_project: str
    wandb_entity: str

    # other options
    sweep_type: str  # bayes, grid, random
    run_cap: int
    folds_idxs: list[int] = field(
        default_factory=lambda: [0]
    )  # sublist of [0, 2, 4, 6, 8]

    # GPU options can be added here


search_space_by_model_name: dict[MLModelNames, dict] = {
    MLModelNames.LOGISTIC_REGRESSION: {
        "model": {
            "parameters": {
                "model_params": {
                    "parameters": {
                        # pipeline params
                        "sklearn_pipeline_param_clf__C": {
                            "values": [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
                        },
                        "sklearn_pipeline_param_clf__fit_intercept": {"values": [True]},
                        "sklearn_pipeline_param_clf__penalty": {"values": ["l2", None]},
                        "sklearn_pipeline_param_clf__solver": {"values": ["lbfgs"]},
                        "sklearn_pipeline_param_clf__random_state": {"values": [1]},
                        "sklearn_pipeline_param_clf__max_iter": {"values": [1000]},
                        "sklearn_pipeline_param_clf__class_weight": {
                            "values": ["balanced", None]
                        },
                        # scaler params
                        "sklearn_pipeline_param_scaler__with_mean": {"values": [True]},
                        "sklearn_pipeline_param_scaler__with_std": {"values": [True]},
                    },
                },
            },
        }
    },
    MLModelNames.KNN: {
        "model": {
            "parameters": {
                "model_params": {
                    "parameters": {
                        # pipeline params
                        "sklearn_pipeline_param_clf__n_neighbors": {
                            "values": [1, 3, 5, 10, 15, 20, 25, 30]
                        },
                        "sklearn_pipeline_param_clf__weights": {
                            "values": ["uniform", "distance"]
                        },
                        "sklearn_pipeline_param_clf__algorithm": {"values": ["auto"]},
                        "sklearn_pipeline_param_clf__leaf_size": {"values": [30]},
                        "sklearn_pipeline_param_clf__p": {"values": [1, 2, 3, 4, 5, 6]},
                        "sklearn_pipeline_param_clf__metric": {"values": ["minkowski"]},
                        # scaler params
                        "sklearn_pipeline_param_scaler__with_mean": {"values": [True]},
                        "sklearn_pipeline_param_scaler__with_std": {"values": [True]},
                    },
                },
            },
        }
    },
    MLModelNames.DUMMY_CLASSIFIER: {
        "model": {
            "parameters": {
                "model_params": {
                    "parameters": {
                        # pipeline params
                        "sklearn_pipeline_param_clf__strategy": {
                            "values": [
                                "stratified",
                                "most_frequent",
                                "prior",
                                "uniform",
                            ]
                        },
                    },
                },
            },
        }
    },
    MLModelNames.SVM: {
        "model": {
            "parameters": {
                "model_params": {
                    "parameters": {
                        # pipeline params
                        "sklearn_pipeline_param_clf__C": {
                            "values": [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
                        },
                        "sklearn_pipeline_param_clf__kernel": {"values": ["rbf"]},
                        "sklearn_pipeline_param_clf__degree": {"values": [3]},
                        "sklearn_pipeline_param_clf__gamma": {
                            "values": ["scale", "auto", 0.1, 0.01, 0.001, 0.0001]
                        },
                        "sklearn_pipeline_param_clf__coef0": {"values": [0.0]},
                        "sklearn_pipeline_param_clf__shrinking": {"values": [True]},
                        "sklearn_pipeline_param_clf__probability": {"values": [False]},
                        "sklearn_pipeline_param_clf__tol": {"values": [0.001]},
                        "sklearn_pipeline_param_clf__class_weight": {
                            "values": ["balanced", None]
                        },
                        # scaler params
                        "sklearn_pipeline_param_scaler__with_mean": {"values": [True]},
                        "sklearn_pipeline_param_scaler__with_std": {"values": [True]},
                    },
                },
            },
        },
    },
}

features_search_space = {
    "model": {
        "parameters": {
            "model_params": {
                "parameters": {
                    "use_DWELL_TIME_WEIGHTED_PARAGRAPH_DOT_QUESTION_CLS_NO_CONTEXT": {
                        "values": [True, False]
                    },
                    "use_QUESTION_RELEVANCE_SPAN_DOT_IA_DWELL_NO_CONTEXT": {
                        "values": [True, False]
                    },
                }
            }
        }
    }
}

cond_pred_run_configs: dict[str, MLRunConfig] = {
    "LogisticRegressionCondPredDavidILFMLArgs": MLRunConfig(
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_variant="LogisticRegressionCondPredDavidILFMLArgs",
        data_variant="NoReread",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "LogisticRegressionCondPredLennaILFMLArgs": MLRunConfig(
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_variant="LogisticRegressionCondPredLennaILFMLArgs",
        data_variant="NoReread",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "LogisticRegressionCondPredDianeILFMLArgs": MLRunConfig(
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_variant="LogisticRegressionCondPredDianeILFMLArgs",
        data_variant="NoReread",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "LogisticRegressionCondPredReadingTimeMLArgs": MLRunConfig(
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_variant="LogisticRegressionCondPredReadingTimeMLArgs",
        data_variant="NoReread",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    # Dummy Classifier
    "DummyClassifierCondPredMLArgs": MLRunConfig(
        model_name=MLModelNames.DUMMY_CLASSIFIER,
        model_variant="DummyClassifierCondPredMLArgs",
        data_variant="NoReread",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    # K-Nearest Neighbors
    "KNearestNeighborsCondPredDavidILFMLArgs": MLRunConfig(
        model_name=MLModelNames.KNN,
        model_variant="KNearestNeighborsCondPredDavidILFMLArgs",
        data_variant="NoReread",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "KNearestNeighborsCondPredLennaILFMLArgs": MLRunConfig(
        model_name=MLModelNames.KNN,
        model_variant="KNearestNeighborsCondPredLennaILFMLArgs",
        data_variant="NoReread",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "KNearestNeighborsCondPredDianeILFMLArgs": MLRunConfig(
        model_name=MLModelNames.KNN,
        model_variant="KNearestNeighborsCondPredDianeILFMLArgs",
        data_variant="NoReread",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "KNearestNeighborsCondPredReadingTimeMLArgs": MLRunConfig(
        model_name=MLModelNames.KNN,
        model_variant="KNearestNeighborsCondPredReadingTimeMLArgs",
        data_variant="NoReread",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    # Support Vector Machine
    "SupportVectorMachineCondPredDavidILFMLArgs": MLRunConfig(
        model_name=MLModelNames.SVM,
        model_variant="SupportVectorMachineCondPredDavidILFMLArgs",
        data_variant="NoReread",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "SupportVectorMachineCondPredLennaILFMLArgs": MLRunConfig(
        model_name=MLModelNames.SVM,
        model_variant="SupportVectorMachineCondPredLennaILFMLArgs",
        data_variant="NoReread",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "SupportVectorMachineCondPredDianeILFMLArgs": MLRunConfig(
        model_name=MLModelNames.SVM,
        model_variant="SupportVectorMachineCondPredDianeILFMLArgs",
        data_variant="NoReread",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "SupportVectorMachineCondPredReadingTimeILFMLArgs": MLRunConfig(
        model_name=MLModelNames.SVM,
        model_variant="SupportVectorMachineCondPredReadingTimeILFMLArgs",
        data_variant="NoReread",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
}

is_correct_run_configs: dict[str, MLRunConfig] = {
    # Logistic Regression
    ## Hunting
    "LRIsCorrectPredDavidILFMLArgsHunting": MLRunConfig(
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_variant="LogisticRegressionIsCorrectPredDavidILFMLArgs",
        data_variant="Hunting",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "LRIsCorrectPredLennaILFMLArgsHunting": MLRunConfig(
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_variant="LogisticRegressionIsCorrectPredLennaILFMLArgs",
        data_variant="Hunting",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "LRIsCorrectPredDianeILFMLArgsHunting": MLRunConfig(
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_variant="LogisticRegressionIsCorrectPredDianeILFMLArgs",
        data_variant="Hunting",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "LRIsCorrectPredReadingTimeMLArgsHunting": MLRunConfig(
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_variant="LogisticRegressionIsCorrectPredReadingTimeMLArgs",
        data_variant="Hunting",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    ## Gathering
    "LRIsCorrectPredDavidILFMLArgsGathering": MLRunConfig(
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_variant="LogisticRegressionIsCorrectPredDavidILFMLArgs",
        data_variant="Gathering",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "LRIsCorrectPredLennaILFMLArgsGathering": MLRunConfig(
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_variant="LogisticRegressionIsCorrectPredLennaILFMLArgs",
        data_variant="Gathering",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "LRIsCorrectPredDianeILFMLArgsGathering": MLRunConfig(
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_variant="LogisticRegressionIsCorrectPredDianeILFMLArgs",
        data_variant="Gathering",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "LRIsCorrectPredReadingTimeMLArgsGathering": MLRunConfig(
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_variant="LogisticRegressionIsCorrectPredReadingTimeMLArgs",
        data_variant="Gathering",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    # Dummy Classifier
    ## Hunting
    "DCIsCorrectPredMLArgsHunting": MLRunConfig(
        model_name=MLModelNames.DUMMY_CLASSIFIER,
        model_variant="DummyClassifierIsCorrectPredMLArgs",
        data_variant="Hunting",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    ## Gathering
    "DCIsCorrectPredMLArgsGathering": MLRunConfig(
        model_name=MLModelNames.DUMMY_CLASSIFIER,
        model_variant="DummyClassifierIsCorrectPredMLArgs",
        data_variant="Gathering",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    # K-Nearest Neighbors
    ## Hunting
    "KNNIsCorPredDavidILFMLArgsHunting": MLRunConfig(
        model_name=MLModelNames.KNN,
        model_variant="KNearestNeighborsIsCorrectPredDavidILFMLArgs",
        data_variant="Hunting",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "KNNIsCorPredLennaILFMLArgsHunting": MLRunConfig(
        model_name=MLModelNames.KNN,
        model_variant="KNearestNeighborsIsCorrectPredLennaILFMLArgs",
        data_variant="Hunting",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "KNNIsCorPredDianeILFMLArgsHunting": MLRunConfig(
        model_name=MLModelNames.KNN,
        model_variant="KNearestNeighborsIsCorrectPredDianeILFMLArgs",
        data_variant="Hunting",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "KNNIsCorPredReadingTimeMLArgsHunting": MLRunConfig(
        model_name=MLModelNames.KNN,
        model_variant="KNearestNeighborsIsCorrectPredReadingTimeMLArgs",
        data_variant="Hunting",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    ## Gathering
    "KNNIsCorPredDavidILFMLArgsGathering": MLRunConfig(
        model_name=MLModelNames.KNN,
        model_variant="KNearestNeighborsIsCorrectPredDavidILFMLArgs",
        data_variant="Gathering",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "KNNIsCorPredLennaILFMLArgsGathering": MLRunConfig(
        model_name=MLModelNames.KNN,
        model_variant="KNearestNeighborsIsCorrectPredLennaILFMLArgs",
        data_variant="Gathering",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "KNNIsCorPredDianeILFMLArgsGathering": MLRunConfig(
        model_name=MLModelNames.KNN,
        model_variant="KNearestNeighborsIsCorrectPredDianeILFMLArgs",
        data_variant="Gathering",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "KNNIsCorPredReadingTimeMLArgsGathering": MLRunConfig(
        model_name=MLModelNames.KNN,
        model_variant="KNearestNeighborsIsCorrectPredReadingTimeMLArgs",
        data_variant="Gathering",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    # Support Vector Machine
    ## Hunting
    "SVMIsCorrectPredDavidILFMLArgsHunting": MLRunConfig(
        model_name=MLModelNames.SVM,
        model_variant="SupportVectorMachineIsCorrectPredDavidILFMLArgs",
        data_variant="Hunting",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "SVMIsCorrectPredLennaILFMLArgsHunting": MLRunConfig(
        model_name=MLModelNames.SVM,
        model_variant="SupportVectorMachineIsCorrectPredLennaILFMLArgs",
        data_variant="Hunting",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "SVMIsCorrectPredDianeILFMLArgsHunting": MLRunConfig(
        model_name=MLModelNames.SVM,
        model_variant="SupportVectorMachineIsCorrectPredDianeILFMLArgs",
        data_variant="Hunting",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "SVMIsCorrectPredReadingTimeMLArgsHunting": MLRunConfig(
        model_name=MLModelNames.SVM,
        model_variant="SupportVectorMachineIsCorrectPredReadingTimeILFMLArgs",
        data_variant="Hunting",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    ## Gathering
    "SVMIsCorrectPredDavidILFMLArgsGathering": MLRunConfig(
        model_name=MLModelNames.SVM,
        model_variant="SupportVectorMachineIsCorrectPredDavidILFMLArgs",
        data_variant="Gathering",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "SVMIsCorrectPredLennaILFMLArgsGathering": MLRunConfig(
        model_name=MLModelNames.SVM,
        model_variant="SupportVectorMachineIsCorrectPredLennaILFMLArgs",
        data_variant="Gathering",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "SVMIsCorrectPredDianeILFMLArgsGathering": MLRunConfig(
        model_name=MLModelNames.SVM,
        model_variant="SupportVectorMachineIsCorrectPredDianeILFMLArgs",
        data_variant="Gathering",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
    "SVMIsCorrectPredReadingTimeILFMLArgsGathering": MLRunConfig(
        model_name=MLModelNames.SVM,
        model_variant="SupportVectorMachineIsCorrectPredReadingTimeILFMLArgs",
        data_variant="Gathering",
        data_path="may05",
        trainer_variant="CfirMLQCond",
        add_feature_search_space=False,
    ),
}

question_pred_run_configs: dict[str, MLRunConfig] = {
    "LogisticRegressionQPredMLArgs": MLRunConfig(
        model_name=MLModelNames.LOGISTIC_REGRESSION,
        model_variant="LogisticRegressionQPredMLArgs",
        data_variant="Hunting",
        data_path="may05",
        trainer_variant="CfirMLQPred",
        add_feature_search_space=True,
    ),
    "DummyClassifierQPredMLArgs": MLRunConfig(
        model_name=MLModelNames.DUMMY_CLASSIFIER,
        model_variant="DummyClassifierQPredMLArgs",
        data_variant="Hunting",
        data_path="may05",
        trainer_variant="CfirMLQPred",
        add_feature_search_space=False,
    ),
    "KNearestNeighborsQPredMLArgs": MLRunConfig(
        model_name=MLModelNames.KNN,
        model_variant="KNearestNeighborsQPredMLArgs",
        data_variant="Hunting",
        data_path="may05",
        trainer_variant="CfirMLQPred",
        add_feature_search_space=True,
    ),
    "SupportVectorMachineQPredMLArgs": MLRunConfig(
        model_name=MLModelNames.SVM,
        model_variant="SupportVectorMachineQPredMLArgs",
        data_variant="Hunting",
        data_path="may05",
        trainer_variant="CfirMLQPred",
        add_feature_search_space=True,
    ),
}

run_configs = {
    **cond_pred_run_configs,
    **is_correct_run_configs,
    **question_pred_run_configs,
    # Add more run configs here
}


def parse_args() -> Tuple[HyperArgs, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweep with a specific config."
    )

    parser.add_argument(
        "--config-name",
        type=str,
        required=True,
        help="Name of the config to use from running_configs.",
    )

    parser.add_argument(
        "--sweep-type",
        type=str,
        default="bayes",
        help="Type of sweep to run. options: bayes, grid, random.",
    )

    parser.add_argument(
        "--run-cap",
        type=int,
        default=30,
        help="Maximum number of runs to execute.",
    )  # not in use when using grid search

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="question_decoding+gathering",
        help="Name of the wandb project to log to.",
    )

    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="lacc-lab",
        help="Name of the wandb entity to log to.",
    )

    cli_args = parser.parse_args()

    print(cli_args)

    return (
        HyperArgs(
            model_name=run_configs[cli_args.config_name]["model_name"],
            model_variant=run_configs[cli_args.config_name]["model_variant"],
            data_variant=run_configs[cli_args.config_name]["data_variant"],
            data_path=run_configs[cli_args.config_name]["data_path"],
            trainer_variant=run_configs[cli_args.config_name]["trainer_variant"],
            wandb_project=cli_args.wandb_project,
            wandb_entity=cli_args.wandb_entity,
            sweep_type=cli_args.sweep_type,
            run_cap=cli_args.run_cap,
            add_feature_search_space=run_configs[cli_args.config_name][
                "add_feature_search_space"
            ],
        ),
        cli_args,
    )


def main():
    hyper_args, cli_args = parse_args()

    # print args
    print(f"Hyper Args:\n{hyper_args}")

    time.sleep(5)

    # run
    search_space = search_space_by_model_name[hyper_args.model_name]

    # add feature search space
    if hyper_args.add_feature_search_space:
        # add features search space
        search_space["model"]["parameters"]["model_params"]["parameters"].update(
            features_search_space["model"]["parameters"]["model_params"]["parameters"]
        )
    sweep_configs = [
        (
            {
                "program": "src/train_ml.py",
                "method": hyper_args.sweep_type,
                "metric": {
                    "goal": "maximize",
                    "name": "balanced_classless_accuracy/val_weighted_average",
                },
                "entity": hyper_args.wandb_entity,
                "project": hyper_args.wandb_project,
                "parameters": search_space,
                "command": [
                    "${env}",
                    "${interpreter}",
                    "${program}",
                    "${args_no_hypens}",
                    f"+model={hyper_args.model_variant}",
                    f"+data={hyper_args.data_variant}",
                    f"+data_path={hyper_args.data_path}",
                    f"+trainer={hyper_args.trainer_variant}",
                    f"data.fold_index={fold_idx}",
                    # f"trainer.devices={hyper_args.gpu_count}",
                    f"trainer.wandb_job_type=hp_s_{cli_args.config_name}",
                ],
            }
            if hyper_args.sweep_type == "grid"
            else {
                "program": "src/train_ml.py",
                "method": hyper_args.sweep_type,
                "metric": {
                    "goal": "maximize",
                    "name": "balanced_classless_accuracy/val_weighted_average",
                },
                "run_cap": hyper_args.run_cap,
                "entity": hyper_args.wandb_entity,
                "project": hyper_args.wandb_project,
                "parameters": search_space,
                "command": [
                    "${env}",
                    "${interpreter}",
                    "${program}",
                    "${args_no_hypens}",
                    f"+model={hyper_args.model_variant}",
                    f"+data={hyper_args.data_variant}",
                    f"+data_path={hyper_args.data_path}",
                    f"+trainer={hyper_args.trainer_variant}",
                    f"data.fold_index={fold_idx}",
                    # f"trainer.devices={hyper_args.gpu_count}",
                    f"trainer.wandb_job_type=hyperparameter_sweep_{hyper_args.model_variant}",
                ],
            }
        )
        for fold_idx in hyper_args.folds_idxs
    ]

    wandb.init(project=hyper_args.wandb_project, entity=hyper_args.wandb_entity)

    sweep_ids = [
        wandb.sweep(
            sweep_config,
            entity=hyper_args.wandb_entity,
            project=hyper_args.wandb_project,
        )
        for sweep_config in sweep_configs
    ]

    # create a bash script per sweep
    time_rn = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for idx, sweep_id in enumerate(sweep_ids):
        filename = f"run_sweep_cfg={cli_args.config_name}_time={time_rn}_s#{idx}.sh"

        bash_script = "#!/bin/bash\n\n"
        bash_script += f'SWEEP_ID="{hyper_args.wandb_entity}/{hyper_args.wandb_project}/{sweep_id}"\n'

        # check if tmux is installed
        bash_script += "if ! command -v tmux &>/dev/null\n"
        bash_script += "then\n"
        bash_script += (
            '    echo "tmux could not be found, please install tmux first."\n'
        )
        bash_script += "    exit 1\n"
        bash_script += "fi\n"

        # add user input requests
        bash_script += 'read -p "Enter GPU number: " GPU_NUM\n'
        bash_script += 'read -p "Enter number of runs on GPU: " RUNS_ON_GPU\n'

        # loop and start tmux
        bash_script += "for ((i=1; i<=RUNS_ON_GPU; i++))\n"
        bash_script += "do\n"
        bash_script += (
            '    session_name="wandb-gpu${GPU_NUM}-dup${i}' + f'-{sweep_id}"\n'
        )
        bash_script += '    tmux new-session -d -s "${session_name}" "CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent ${SWEEP_ID}"\\; set-option -t "${session_name}" remain-on-exit on\n'
        bash_script += '    echo Running: tmux new-session -d -s "${session_name}" "CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent ${SWEEP_ID}"\\; set-option -t "${session_name}" remain-on-exit on\n'
        bash_script += '    echo "Launched W&B agent for GPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}."\n'
        bash_script += "done\n"

        # save sh file
        with open(filename, "w") as f:
            f.write(bash_script)

        # make exec
        st = os.stat(filename)
        os.chmod(filename, st.st_mode | stat.S_IEXEC)


if __name__ == "__main__":
    main()
