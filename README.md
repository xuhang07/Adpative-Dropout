# Adpative-Dropout
Code for "Adaptive Dropout: Unleashing Dropout across Layers for Generalizable Image Super-Resolution"-CVPR2025

# Abstract
Blind Super-Resolution(blind SR) aims to enhance the model's generalization ability with unknown degradation, yet it still encounters severe overfitting issues. Some previous methods inspired by dropout, which enhances generalization by regularizing features, have shown promising results in blind SR. Nevertheless, these methods focus solely on regularizing features before the final layer and overlook the need for generalization in features at intermediate layers. Without explicit regularization of features at intermediate layers, the blind SR network struggles to obtain well-generalized feature representations. However, the key challenge is that directly applying dropout to intermediate layers leads to a significant performance drop, which we attribute to the inconsistency in training-testing and across layers it introduced. Therefore, we propose Adaptive Dropout, a new regularization method for blind SR models, which mitigates the inconsistency and facilitates application across intermediate layers of networks. Specifically, for training-testing inconsistency, we re-design the form of dropout and integrate the features before and after dropout adaptively. For inconsistency in generalization requirements across different layers, we innovatively design an adaptive training strategy to strengthen feature propagation by layer-wise annealing. Experimental results show that our method outperforms all past regularization methods on both synthetic and real-world benchmark datasets, also highly effective in other image restoration tasks.

## Installation

1. Install dependent packages 
- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.7.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install basicsr`
- [option] Python packages: [`pip install tensorboardX`](https://github.com/lanpa/tensorboardX), for visualizing curves.

2. Clone this github repo. 
```
git clone https://github.com/xuhang07/Adpative-Dropout.git
cd Adpative-Dropout
```
## Evaluation

1. Download the testing datasets (Set5, Set14, B100, Manga109, Urban100) and move them to `./dataset/benchmark`.
[Google Drive](https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u).

2. Add degradations to testing datasets.
```
cd ./dataset
python add_degradations.py
```
3. Our pretrained model is in ```Real-train/pretrained_models/best.pth```, for the pretrained model of baseline, please refer to [RDSR](https://github.com/XPixelGroup/RDSR/tree/main) and [Simple-Align](https://github.com/Dreamzz5/Simple-Align)
   
4. Run the testing commands.
```
CUDA_VISIBLE_DEVICES=1 python realesrgan/test.py -opt options/test/test_realsrresnet_withdropout.yml
```
6. The output results will be sorted in `./results`.

## Training
**Some steps require replacing your local paths.**

1. Move to experiment dir.
```
cd Real-train
```

2. Download the training datasets([DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)), move it to `./dataset` and validation dataset(Set5), move it to `./dataset/benchmark`.

3. Run the training commands.
```
cd codes
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 realesrgan/train.py -opt options/train/train_realsrresnet.yml --launcher pytorch --auto_resume
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 realesrgan/train.py -opt options/train/train_realsrresnet_withdropout.yml --launcher pytorch --auto_resume
```
4. The experiments will be sorted in `./experiments`.

### Acknowledgement

Many parts of this code is adapted from:

- [RDSR](https://github.com/XPixelGroup/RDSR/tree/main)
- [BasicSR](https://github.com/XPixelGroup/BasicSR/tree/master)
- [Simple-Align](https://github.com/Dreamzz5/Simple-Align)

We thank the authors for sharing codes for their great works.



