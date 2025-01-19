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