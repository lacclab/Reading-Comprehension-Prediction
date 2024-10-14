# Fine-Grained Prediction of Reading Comprehension from Eye Movements


[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)


## Getting Started

### Prerequisites

- [Mamba](https://github.com/conda-forge/miniforge#mambaforge) or Conda

### Setup

1. **Clone the Repository**

    Start by cloning the repository to your local machine:

    ```bash
    git clone https://github.com/lacclab/Reading-Comprehension-Prediction.git
    cd Reading-Comprehension-Prediction
    ```

2. **Create a Virtual Environment**

    Create a new virtual environment using Mamba (or Conda) and install the dependencies:

    ```bash
    mamba env create -f environment.yaml
    ```


## Usage

### Training

1. **Default Training:**

    Run the training + eval script with default parameters. This will perform training and predictions on the test set in a cross-validation setting:

    ```bash
    python scripts/run_wrapper.py
    ```

2. **Custom Training:**

    Run the training script with custom data, model, and trainer options. You can choose to not perform cross-validation by adding the `--single_run` flag.
    Add `--skip_train` to skip training and only perform predictions on the test set. Add `--skip_eval` to skip evaluation and only perform training.
    If you want the terminal pane to close after the run, add the `--do_not_keep_pane_alive` flag:

    ```bash
    python scripts/run_wrapper.py --data_options "hunting" "gathering" --model_options "roberteye_duplicate_fixation" --trainer "shubi"
    ```

3. **Advanced Training:**

    Run the training script with custom parameters and specify the GPU device for training. You can also override any other parameters defined in `model_args.py`:

    ```bash
    python src/train.py +trainer=shubi +model=roberteye_duplicate_fixation +data=hunting trainer.devices=[1] # and any other overrides
    ```

## Adding a New Model

Follow these steps to add a new model:

1. **Create Model Class:** In the `src/models` directory, create a new Python file. In this file, define a class that inherits from `BaseModel` and implements the following methods:
    - `forward`: This method should define the forward pass of your model.
    - `shared_step`: This method should define the entire forward process. It should call the `forward` method, calculate the loss and metrics, and return `ordered_label, loss, ordered_logits`. See existing models for examples.

2. **Update ModelNames Enum:** Add a new entry to the `ModelNames` enum in `src/configs/enums.py`. This will be the identifier for your model.

3. **Update Model Configurations:** In the `src/configs/model_args.py` file, perform the following steps:
    - Create a new class that inherits from `BaseModelArgs`. This class should (atleast) define any variables that are marked as `MISSING` in the `BaseModelArgs` class.
    - Add your new class to the `register_model_configs` function.

4. **Update ModelMapping Enum:** Add a new entry for your model to the `ModelMapping` enum in `src/configs/config.py`. This will allow your model to be selected based on the configuration.


### Others

* Under `scripts` are additional scrips for running hyperparameter tuning and evaluation.

This project was developed with the assistance of GitHub Copilot, an AI-powered coding assistant. All generated code was carefully reviewed.


## Citation

If you use this repository, please consider citing the following work:

```bibtex
@inproceedings{Shubi2024finegrained,
    title={Fine-Grained Prediction of Reading Comprehension from Eye Movements},
    author={Omer Shubi, Yoav Meiri, Cfir Avraham Hadar, Yevgeni Berzak},
    booktitle={The 2024 Conference on Empirical Methods in Natural Language Processing},
    year={2024},
}
```
