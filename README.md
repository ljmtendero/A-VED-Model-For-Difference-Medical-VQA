# A-VED-Model-For-Difference-Medical-VQA

Official implementation of Unveiling Differences: A Vision Encoder-Decoder Model for Difference Medical Visual Question Answering

![Model Architecture](figures/MODELARQUITECTURE.png)

## Installation

```
conda create -n mdvqa_env python=3.9
conda activate mdvqa_env

pip install vilmedic==1.3.2
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install albumentations
```
If there is any error related to the CV2 package:
```
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless 
pip install opencv-python==4.7.0.72
```

Update the paths.py file from both Swin_Radiology_Finetune and Vision_Encoder_Decoder_MDiffVQA


## Training workflow

Stage 1 - Finetune Swin Transformer

Navigate to the Swin_Radiology_Finetune, you can prepare your data through the "prepare_data.ipynb" notebooks and train a Swin model. Execute the swin_finetune.py to finetune a Swin model on your data and labels, and use swin_finetune_analysis.py to analize the results.

Stage 2 and 3 - Train VED Model

Navigate to Vision_Encoder_Decoder_MDiffVQA, in the train folder you will find a nll_train.sh and a mytrain_nll.py, these two files can be used to train the model for the following stages, you will need to adjust the learning rate and freeze-unfreeze the encoder through that files.

