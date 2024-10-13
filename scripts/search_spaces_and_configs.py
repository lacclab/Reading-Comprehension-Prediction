from dataclasses import dataclass

from run_wrapper import DataOptions, DataPathOptions, ModelOptions, TrainerOptions

from src.configs.constants import BackboneNames, ModelNames

dl_learning_rates = [
    1e-5,
    3e-5,
    1e-4,
]

# backbones = [
#     # BackboneNames.ROBERTA_RACE,
#     # BackboneNames.ROBERTA_LARGE,
#     BackboneNames.ROBERTA_BASE,
# ]  #! NOTE THIS (Commented out in postfusion and mag only)

contrastive_loss = [
    # True,
    False
]

add_answers = [
    # True,
    False
]  #! NOTE THIS (Commented out in postfusion and mag only)

freeze = [
    # True,
    False,
]
dropout = [
    0.1,
    0.3,
    0.5,
]

pf_att_mask = [
    True,
    # False,
]

search_space_by_model_name: dict[ModelNames, dict] = {
    ModelNames.ROBERTEYE_MODEL: {
        "model": {
            "parameters": {
                "model_params": {
                    "parameters": {
                        "eye_projection_dropout": {"values": dropout},
                    }
                },
                "freeze": {
                    "values": freeze,
                },
                # "backbone": {"values": backbones},
                "add_contrastive_loss": {
                    "values": contrastive_loss,
                },
                # "prediction_config": {
                #     "parameters": {"add_answers": {"values": add_answers}}
                # },
            }
        },
        "trainer": {
            "parameters": {
                "learning_rate": {"values": dl_learning_rates},
            }
        },
    },  # * should be run with --run-cap 108 (2*3*2*3*3)
    ModelNames.BEYELSTM_MODEL: {
        "model": {
            "parameters": {
                "add_contrastive_loss": {
                    "values": contrastive_loss,
                },
                "model_params": {
                    "parameters": {
                        "dropout_rate": {
                            "values": dropout,
                        },
                        "embedding_dim": {
                            "values": [4, 8],
                        },
                        "lstm_block_fc1_out_dim": {
                            "values": [50],
                        },
                        "lstm_block_fc2_out_dim": {
                            "values": [20],
                        },
                        "gsf_out_dim": {
                            "values": [32],
                        },
                        "after_cat_fc_hidden_dim": {
                            "values": [32],
                        },
                        "hidden_dim": {
                            "values": [64, 128],
                        },
                    }
                },
            }
        },
        "trainer": {
            "parameters": {
                "learning_rate": {"values": [0.001, 0.003, 0.01]},
            }
        },
    },  # * should be run with --run-cap 30
    ModelNames.AHN_CNN_MODEL: {
        "model": {
            "parameters": {
                "model_params": {
                    "parameters": {
                        "hidden_dim": {"values": [40]},
                        "conv_kernel_size": {"values": [3]},
                        "pooling_kernel_size": {"values": [2]},
                        "fc_hidden_dim1": {"values": [50]},
                        "fc_hidden_dim2": {"values": [20]},
                        "fc_dropout": {"values": dropout},
                    }
                }
            }
        },
        "trainer": {
            "parameters": {
                "learning_rate": {"values": [1e-3] + dl_learning_rates},
            }
        },
    },
    ModelNames.AHN_RNN_MODEL: {
        "model": {
            "parameters": {
                "model_params": {
                    "parameters": {
                        "hidden_dim": {"values": [25]},
                        "num_lstm_layers": {"values": [1]},
                        "fc_hidden_dim": {"values": [2]},
                        "fc_dropout": {"values": [0.3]},
                    }
                }
            }
        }
    },
    ModelNames.MAG_MODEL: {
        "model": {
            "parameters": {
                "model_params": {
                    "parameters": {
                        "mag_dropout": {"values": dropout},
                        "mag_injection_index": {  # Starts from 0 and should be no more than max number of layers. If roberta-base and >13 cutoff in train.py
                            "values": [
                                0,
                                11,
                                23,
                            ],
                        },
                    }
                },
                # "backbone": {"values": backbones},
                "add_contrastive_loss": {
                    "values": contrastive_loss,
                },
                # "prediction_config": {
                #     "parameters": {"add_answers": {"values": add_answers}}
                # },
            }
        },
        "trainer": {
            "parameters": {
                "learning_rate": {"values": dl_learning_rates},
            }
        },
    },  # * should be run with --run-cap (3*3*3*2*5 = 270)
    ModelNames.POSTFUSION_MODEL: {
        "model": {
            "parameters": {
                "model_params": {
                    "parameters": {
                        # "cross_attention_dropout": {"values": dropout},
                        "eye_projection_dropout": {"values": dropout},
                        "use_attn_mask": {
                            "values": pf_att_mask,
                        },
                    }
                },
                # "backbone": {"values": backbones},
                # "freeze": {
                #    "values": freeze,
                # },
                "add_contrastive_loss": {
                    "values": contrastive_loss,
                },
                # "prediction_config": {
                #     "parameters": {"add_answers": {"values": add_answers}}
                # },
            }
        },
        "trainer": {
            "parameters": {
                "learning_rate": {"values": dl_learning_rates},
            }
        },
    },  # * should be run with --run-cap 128
    ModelNames.EYETTENTION_MODEL: {
        "trainer": {
            "parameters": {
                "learning_rate": {"values": [1e-3] + dl_learning_rates},
            }
        },
        "model": {
            "parameters": {
                "model_params": {
                    "parameters": {
                        # "num_LSTM_layers": {"values": [2, 4, 8]},
                        "LSTM_dropout": {"values": [0.2] + dropout},
                    }
                }
            }
        },
    },
}


@dataclass
class RunConfig:
    model_name: ModelNames
    model_variant: ModelOptions
    data_variant: DataOptions
    data_path: DataPathOptions
    trainer_variant: TrainerOptions


@dataclass
class IsCorrectSampling(RunConfig):
    data_path: DataPathOptions = "may05"
    trainer_variant: TrainerOptions = "IsCorrectSampling"


HUNTING = "Hunting"
GATHERING = "Gathering"

run_configs: dict[str, RunConfig] = {
    # "RoBERTeyeQCondDuplicate": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeQCondDuplicate",
    #     data_variant="NoReread",
    #     data_path="may05",
    #     trainer_variant="CfirRoBERTaEye",
    # ),
    # "RoBERTeyeQCondConcat": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeQCondConcat",
    #     data_variant="NoReread",
    #     data_path="may05",
    #     trainer_variant="CfirRoBERTaEye",
    # ),
    # "BEyeLSTMQCond": RunConfig(
    #     model_name=ModelNames.BEYELSTM_MODEL,
    #     model_variant="BEyeLSTMQCond",
    #     data_variant="NoReread",
    #     data_path="may05",
    #     trainer_variant="CfirBEyeLSTM",
    # ),
    # "MAGQCondPredDuplicateNoFix": RunConfig(
    #     model_name=ModelNames.MAG_MODEL,
    #     model_variant="MAGQCondPredDuplicateNoFix",
    #     data_variant="NoReread",
    #     data_path="may05",
    #     trainer_variant="CfirMAG",
    # ),
    # # Condition Prediction:
    # "RoBERTeyeCondPredConcat": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeCondPredConcat",
    #     data_variant="NoReread",
    #     data_path="may05",
    #     trainer_variant="CfirRoBERTaEye",
    # ),
    # "MAGCondPredConcatNoFix": RunConfig(
    #     model_name=ModelNames.MAG_MODEL,
    #     model_variant="MAGCondPredConcatNoFix",
    #     data_variant="NoReread",
    #     data_path="may05",
    #     trainer_variant="CfirMAG",
    # ),
    # # Question Decoding:
    # "RoBERTeyeQPredConcatNoFix": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeQPredConcatNoFix",
    #     data_variant="Hunting",
    #     data_path="may05",
    #     trainer_variant="CfirRoBERTaEye",
    # ),
    # "RoBERTeyeQPredDuplicateNoFix": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeQPredDuplicateNoFix",
    #     data_variant="Hunting",
    #     data_path="may05",
    #     trainer_variant="CfirRoBERTaEye",
    # ),
    # ###### Shubi ########
    # "RoBERTeyeConcatIAReadingCompH": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeConcatIAReadingComp",
    #     data_variant="Hunting",
    #     data_path="may05",
    #     trainer_variant="ReadingCompBase",
    # ),
    # "RoBERTeyeConcatIAReadingCompG": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeConcatIAReadingComp",
    #     data_variant="Gathering",
    #     data_path="may05",
    #     trainer_variant="ReadingCompBase",
    # ),
    # "RoBERTeyeConcatFixationReadingCompH": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeConcatFixationReadingComp",
    #     data_variant="Hunting",
    #     data_path="may05",
    #     trainer_variant="ReadingCompBase",
    # ),
    # "RoBERTeyeConcatFixationReadingCompG": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeConcatFixationReadingComp",
    #     data_variant="Gathering",
    #     data_path="may05",
    #     trainer_variant="ReadingCompBase",
    # ),
    # "MAGConcatReadingCompH": RunConfig(
    #     model_name=ModelNames.MAG_MODEL,
    #     model_variant="MAGConcatReadingComp",
    #     data_variant="Hunting",
    #     data_path="may05",
    #     trainer_variant="ReadingCompBase",
    # ),
    # "MAGConcatReadingCompG": RunConfig(
    #     model_name=ModelNames.MAG_MODEL,
    #     model_variant="MAGConcatReadingComp",
    #     data_variant="Gathering",
    #     data_path="may05",
    #     trainer_variant="ReadingCompBase",
    # ),
    # "RoBERTaNoEyesH": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTaNoEyes",
    #     data_variant="Hunting",
    #     data_path="may05",
    #     trainer_variant="ReadingCompBase",
    # ),
    # "RoBERTaNoEyesG": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTaNoEyes",
    #     data_variant="Gathering",
    #     data_path="may05",
    #     trainer_variant="ReadingCompBase",
    # ),
    # "PostFusionReadingCompH": RunConfig(
    #     model_name=ModelNames.POSTFUSION_MODEL,
    #     model_variant="PostFusionReadingComp",
    #     data_variant="Hunting",
    #     data_path="may05",
    #     trainer_variant="ReadingCompBase",
    # ),
    # "PostFusionReadingCompG": RunConfig(
    #     model_name=ModelNames.POSTFUSION_MODEL,
    #     model_variant="PostFusionReadingComp",
    #     data_variant="Gathering",
    #     data_path="may05",
    #     trainer_variant="ReadingCompBase",
    # ),  ###### Shubi IsCorrect ##########
    # "RoBERTeyeConcatIAIsCorrectH": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeConcatIAIsCorrect",
    #     data_variant="Hunting",
    #     data_path="may05",
    #     trainer_variant="ReadingCompBase",
    # ),
    # "RoBERTeyeConcatIAIsCorrectG": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeConcatIAIsCorrect",
    #     data_variant="Gathering",
    #     data_path="may05",
    #     trainer_variant="ReadingCompBase",
    # ),
    # "RoBERTeyeConcatIAIsCorrectCLH": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeConcatIAIsCorrectCL",
    #     data_variant="Hunting",
    #     data_path="may05",
    #     trainer_variant="ReadingCompBase",
    # ),
    # "RoBERTeyeConcatIAIsCorrectCLG": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeConcatIAIsCorrectCL",
    #     data_variant="Gathering",
    #     data_path="may05",
    #     trainer_variant="ReadingCompBase",
    # ),
    # "RoBERTeyeConcatNoEyesIsCorrectH": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeConcatNoEyesIsCorrect",
    #     data_variant="Hunting",
    #     data_path="may05",
    #     trainer_variant="ReadingCompBase",
    # ),
    # "RoBERTeyeConcatNoEyesIsCorrectG": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeConcatNoEyesIsCorrect",
    #     data_variant="Gathering",
    #     data_path="may05",
    #     trainer_variant="ReadingCompBase",
    # ),
    # "RoBERTeyeConcatIAIsCorrectSamplingCLossH": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeConcatIAIsCorrectCLSampling",
    #     data_variant="Hunting",
    #     data_path="may05",
    #     trainer_variant="IsCorrectSampling",
    # ),
    # "RoBERTeyeConcatIAIsCorrectSamplingCLossG": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeConcatIAIsCorrectCLSampling",
    #     data_variant="Gathering",
    #     data_path="may05",
    #     trainer_variant="IsCorrectSampling",
    # ),
    # "RoBERTeyeConcatFixationIsCorrectSamplingCLossH": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeConcatFixationIsCorrectCLSampling",
    #     data_variant="Hunting",
    #     data_path="may05",
    #     trainer_variant="IsCorrectSampling",
    # ),
    # "RoBERTeyeConcatFixationIsCorrectSamplingCLossG": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeConcatFixationIsCorrectCLSampling",
    #     data_variant="Gathering",
    #     data_path="may05",
    #     trainer_variant="IsCorrectSampling",
    # ),
    # "PostFusionConcatIsCorrectSamplingCLossH": RunConfig(
    #     model_name=ModelNames.POSTFUSION_MODEL,
    #     model_variant="PostFusionConcatIsCorrectCLSampling",
    #     data_variant="Hunting",
    #     data_path="may05",
    #     trainer_variant="IsCorrectSampling",
    # ),
    # "PostFusionConcatIsCorrectSamplingCLossG": RunConfig(
    #     model_name=ModelNames.POSTFUSION_MODEL,
    #     model_variant="PostFusionConcatIsCorrectCLSampling",
    #     data_variant="Gathering",
    #     data_path="may05",
    #     trainer_variant="IsCorrectSampling",
    # ),
    # "MAGConcatIsCorrectSamplingCLossH": RunConfig(
    #     model_name=ModelNames.MAG_MODEL,
    #     model_variant="MAGConcatIsCorrectCLSampling",
    #     data_variant="Hunting",
    #     data_path="may05",
    #     trainer_variant="IsCorrectSampling",
    # ),
    # "MAGConcatIsCorrectSamplingCLossG": RunConfig(
    #     model_name=ModelNames.MAG_MODEL,
    #     model_variant="MAGConcatIsCorrectCLSampling",
    #     data_variant="Gathering",
    #     data_path="may05",
    #     trainer_variant="IsCorrectSampling",
    # ),
    # "RoBERTeyeConcatNoEyesIsCorrectSamplingCLossH": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeConcatNoEyesIsCorrectCLSampling",
    #     data_variant="Hunting",
    #     data_path="may05",
    #     trainer_variant="IsCorrectSampling",
    # ),
    # "RoBERTeyeConcatNoEyesIsCorrectSamplingG": RunConfig(
    #     model_name=ModelNames.ROBERTEYE_MODEL,
    #     model_variant="RoBERTeyeConcatNoEyesIsCorrectSampling",
    #     data_variant="Gathering",
    #     data_path="may05",
    #     trainer_variant="IsCorrectSampling",
    # ),
    "MAGG": IsCorrectSampling(
        model_name=ModelNames.MAG_MODEL,
        model_variant="MAG",
        data_variant=GATHERING,
    ),
    "MAGH": IsCorrectSampling(
        model_name=ModelNames.MAG_MODEL,
        model_variant="MAG",
        data_variant=HUNTING,
    ),
    "PostFusionG": IsCorrectSampling(
        model_name=ModelNames.POSTFUSION_MODEL,
        model_variant="PostFusion",
        data_variant=GATHERING,
    ),
    "PostFusionH": IsCorrectSampling(
        model_name=ModelNames.POSTFUSION_MODEL,
        model_variant="PostFusion",
        data_variant=HUNTING,
    ),
    "RoberteyeWordG": IsCorrectSampling(
        model_name=ModelNames.ROBERTEYE_MODEL,
        model_variant="RoberteyeWord",
        data_variant=GATHERING,
    ),
    "RoberteyeWordH": IsCorrectSampling(
        model_name=ModelNames.ROBERTEYE_MODEL,
        model_variant="RoberteyeWord",
        data_variant=HUNTING,
    ),
    "RoberteyeFixationG": IsCorrectSampling(
        model_name=ModelNames.ROBERTEYE_MODEL,
        model_variant="RoberteyeFixation",
        data_variant=GATHERING,
    ),
    "RoberteyeFixationH": IsCorrectSampling(
        model_name=ModelNames.ROBERTEYE_MODEL,
        model_variant="RoberteyeFixation",
        data_variant=HUNTING,
    ),
    "AhnCNNG": IsCorrectSampling(
        model_name=ModelNames.AHN_CNN_MODEL,
        model_variant="AhnCNN",
        data_variant=GATHERING,
        trainer_variant="Ahn",
    ),
    "AhnCNNH": IsCorrectSampling(
        model_name=ModelNames.AHN_CNN_MODEL,
        model_variant="AhnCNN",
        data_variant=HUNTING,
        trainer_variant="Ahn",
    ),
    "RobertaG": IsCorrectSampling(
        model_name=ModelNames.ROBERTEYE_MODEL,
        model_variant="Roberta",
        data_variant=GATHERING,
    ),
    "RobertaH": IsCorrectSampling(
        model_name=ModelNames.ROBERTEYE_MODEL,
        model_variant="Roberta",
        data_variant=HUNTING,
    ),
    "BEyeLSTMG": RunConfig(
        model_name=ModelNames.BEYELSTM_MODEL,
        model_variant="BEyeLSTMArgs",
        data_variant=GATHERING,
        data_path="may05",
        trainer_variant="BEyeLSTM",
    ),
    "BEyeLSTMH": RunConfig(
        model_name=ModelNames.BEYELSTM_MODEL,
        model_variant="BEyeLSTMArgs",
        data_variant=HUNTING,
        data_path="may05",
        trainer_variant="BEyeLSTM",
    ),
    "EyettentionG": RunConfig(
        model_name=ModelNames.EYETTENTION_MODEL,
        model_variant="Eyettention",
        data_variant=GATHERING,
        data_path="may05",
        trainer_variant="Eyettention",
    ),
    "EyettentionH": RunConfig(
        model_name=ModelNames.EYETTENTION_MODEL,
        model_variant="Eyettention",
        data_variant=HUNTING,
        data_path="may05",
        trainer_variant="Eyettention",
    ),
    ## Extended task
    "PostFusionAnswersG": IsCorrectSampling(
        model_name=ModelNames.POSTFUSION_MODEL,
        model_variant="PostFusionAnswers",
        data_variant=GATHERING,
    ),
    "PostFusionAnswersH": IsCorrectSampling(
        model_name=ModelNames.POSTFUSION_MODEL,
        model_variant="PostFusionAnswers",
        data_variant=HUNTING,
    ),
    "PostFusionMultiClassG": IsCorrectSampling(
        model_name=ModelNames.POSTFUSION_MODEL,
        model_variant="PostFusionMultiClass",
        data_variant=GATHERING,
    ),
    "PostFusionMultiClassH": IsCorrectSampling(
        model_name=ModelNames.POSTFUSION_MODEL,
        model_variant="PostFusionMultiClass",
        data_variant=HUNTING,
    ),
    "PostFusionAnswersMultiClassG": IsCorrectSampling(
        model_name=ModelNames.POSTFUSION_MODEL,
        model_variant="PostFusionAnswersMultiClass",
        data_variant=GATHERING,
    ),
    "PostFusionAnswersMultiClassH": IsCorrectSampling(
        model_name=ModelNames.POSTFUSION_MODEL,
        model_variant="PostFusionAnswersMultiClass",
        data_variant=HUNTING,
    ),
    "PostFusionSelectedAnswersMultiClassG": IsCorrectSampling(
        model_name=ModelNames.POSTFUSION_MODEL,
        model_variant="PostFusionSelectedAnswersMultiClass",
        data_variant=GATHERING,
    ),
    "PostFusionSelectedAnswersMultiClassH": IsCorrectSampling(
        model_name=ModelNames.POSTFUSION_MODEL,
        model_variant="PostFusionSelectedAnswersMultiClass",
        data_variant=HUNTING,
    ),
    "RobertaSelectedAnswersMultiClassG": IsCorrectSampling(
        model_name=ModelNames.ROBERTEYE_MODEL,
        model_variant="RobertaSelectedAnswersMultiClass",
        data_variant=GATHERING,
    ),
    "RobertaSelectedAnswersMultiClassH": IsCorrectSampling(
        model_name=ModelNames.ROBERTEYE_MODEL,
        model_variant="RobertaSelectedAnswersMultiClass",
        data_variant=HUNTING,
    ),
    "MAGSelectedAnswersMultiClassG": IsCorrectSampling(
        model_name=ModelNames.MAG_MODEL,
        model_variant="MAGSelectedAnswersMultiClass",
        data_variant=GATHERING,
    ),
    "MAGSelectedAnswersMultiClassH": IsCorrectSampling(
        model_name=ModelNames.MAG_MODEL,
        model_variant="MAGSelectedAnswersMultiClass",
        data_variant=HUNTING,
    ),
    ##### Postsubmission
    "MAGRaceG": IsCorrectSampling(
        model_name=ModelNames.MAG_MODEL,
        model_variant="MAG",
        data_variant=GATHERING,
    ),
    "MAGRaceH": IsCorrectSampling(
        model_name=ModelNames.MAG_MODEL,
        model_variant="MAG",
        data_variant=HUNTING,
    ),
    "MAGBaseG": IsCorrectSampling(
        model_name=ModelNames.MAG_MODEL,
        model_variant="MAGBase",
        data_variant=GATHERING,
    ),
    "MAGBaseH": IsCorrectSampling(
        model_name=ModelNames.MAG_MODEL,
        model_variant="MAGBase",
        data_variant=HUNTING,
    ),
    "MAGFreezeG": IsCorrectSampling(
        model_name=ModelNames.MAG_MODEL,
        model_variant="MAGFreeze",
        data_variant=GATHERING,
    ),
    "MAGFreezeH": IsCorrectSampling(
        model_name=ModelNames.MAG_MODEL,
        model_variant="MAGFreeze",
        data_variant=HUNTING,
    ),
    "MAGWordsG": IsCorrectSampling(
        model_name=ModelNames.MAG_MODEL,
        model_variant="MAGWords",
        data_variant=GATHERING,
    ),
    "MAGWordsH": IsCorrectSampling(
        model_name=ModelNames.MAG_MODEL,
        model_variant="MAGWords",
        data_variant=HUNTING,
    ),
    "MAGEyesG": IsCorrectSampling(
        model_name=ModelNames.MAG_MODEL,
        model_variant="MAGEyes",
        data_variant=GATHERING,
    ),
    "MAGEyesH": IsCorrectSampling(
        model_name=ModelNames.MAG_MODEL,
        model_variant="MAGEyes",
        data_variant=HUNTING,
    ),
    "PostFusionFreezeG": IsCorrectSampling(
        model_name=ModelNames.POSTFUSION_MODEL,
        model_variant="PostFusionFreeze",
        data_variant=GATHERING,
    ),
    "PostFusionFreezeH": IsCorrectSampling(
        model_name=ModelNames.POSTFUSION_MODEL,
        model_variant="PostFusionFreeze",
        data_variant=HUNTING,
    ),  #### Last minute
    "RoberteyeWordLingG": IsCorrectSampling(
        model_name=ModelNames.ROBERTEYE_MODEL,
        model_variant="RoberteyeWordLing",
        data_variant=GATHERING,
    ),
    "RoberteyeWordLingH": IsCorrectSampling(
        model_name=ModelNames.ROBERTEYE_MODEL,
        model_variant="RoberteyeWordLing",
        data_variant=HUNTING,
    ),
    "RoberteyeWordEyesG": IsCorrectSampling(
        model_name=ModelNames.ROBERTEYE_MODEL,
        model_variant="RoberteyeWordEyes",
        data_variant=GATHERING,
    ),
    "RoberteyeWordEyesH": IsCorrectSampling(
        model_name=ModelNames.ROBERTEYE_MODEL,
        model_variant="RoberteyeWordEyes",
        data_variant=HUNTING,
    ),
    "MAGSelectedAnswersMultiClassLingG": IsCorrectSampling(
        model_name=ModelNames.MAG_MODEL,
        model_variant="MAGSelectedAnswersMultiClassLing",
        data_variant=GATHERING,
    ),
    "MAGSelectedAnswersMultiClassLingH": IsCorrectSampling(
        model_name=ModelNames.MAG_MODEL,
        model_variant="MAGSelectedAnswersMultiClassLing",
        data_variant=HUNTING,
    ),
    "RoberteyeWordSelectedAnswersMultiClassG": IsCorrectSampling(
        model_name=ModelNames.ROBERTEYE_MODEL,
        model_variant="RoberteyeWordSelectedAnswersMultiClass",
        data_variant=GATHERING,
    ),
    "RoberteyeWordSelectedAnswersMultiClassH": IsCorrectSampling(
        model_name=ModelNames.ROBERTEYE_MODEL,
        model_variant="RoberteyeWordSelectedAnswersMultiClass",
        data_variant=HUNTING,
    ),
    "RoberteyeWordLingSelectedAnswersMultiClassG": IsCorrectSampling(
        model_name=ModelNames.ROBERTEYE_MODEL,
        model_variant="RoberteyeWordLingSelectedAnswersMultiClass",
        data_variant=GATHERING,
    ),
    "RoberteyeWordLingSelectedAnswersMultiClassH": IsCorrectSampling(
        model_name=ModelNames.ROBERTEYE_MODEL,
        model_variant="RoberteyeWordLingSelectedAnswersMultiClass",
        data_variant=HUNTING,
    ),
    "RoberteyeFixationSelectedAnswersMultiClassG": IsCorrectSampling(
        model_name=ModelNames.ROBERTEYE_MODEL,
        model_variant="RoberteyeFixationSelectedAnswersMultiClass",
        data_variant=GATHERING,
    ),
    "RoberteyeFixationSelectedAnswersMultiClassH": IsCorrectSampling(
        model_name=ModelNames.ROBERTEYE_MODEL,
        model_variant="RoberteyeFixationSelectedAnswersMultiClass",
        data_variant=HUNTING,
    ),
}
