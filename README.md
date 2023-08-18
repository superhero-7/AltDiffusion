# AltDiffuson
Source code for paper: "AltDiffusion: A multilingual Text-to-Image diffusion model"

# Introduction




## Project Structure

### 1.1 ckpt

Storing ckpt for different models

### 1.2 src

    Contains the main model, training code;

#### 1.2.1 callbacks

        Contains various log implementations, inserted through callback functions during training;

#### 1.2.2 configs

        Training and reasoning profiles

#### 1.2.3 ldm

        The body code for stable diffusion is all here

#### 1.2.4 lm

        The model code for altclip is all here

#### 1.2.5 scripts

        Training startup code

Dataclass.py： Data loading class

lr_scheduler.py: Setting of learning rate

1.3 misc

Includes data preprocessing, model inference, evaluation, and more; (dirty folder)


## Environment Configuration

```python
pip install torch 1.12.1 torchvision 0.13.1
cd src
pip install -r requirements.txt
```

## Training

The training startup script is located in src/scripts/run_multinode.sh. This script is not subject to change. The code path and saved log.txt path should be changed when first used.

The training configuration parameter Settings are located in /src/configs/train_multi.yaml, where all parameters that need to be modified are located.

Therefore, the training operation process is as follows:



1. Modify training configuration parameters, such as learning rate, data, etc.;

2. Run the command to fill in:

```python
        bash your_codepath_to_altdiffusion/src/scripts/run_multinode.sh
```

## Inference

Inference scripts are located at misc/nb/inference.ipynb, and simply replace the opt.ckpt in them to test different models

## Evaluation

/misc/evaluation is the source code for MS-COCO evaluation, including translate script, generation script, and metrics calculation script.

/misc/evaluation_new is the source code for MG-18 evaluation(mentioned in the paper), including translate script, generation script, and metrics calculation script.

/misc/human_evaluaiton is the source code for MC-18 evaluation(mentioned in the paper), including translate script,
evaluation interface.
