# ACRE
This repo contains code for our CVPR 2021 paper.

[ACRE: <u>A</u>bstract <u>C</u>ausal <u>RE</u>asoning Beyond Covariation](http://wellyzhang.github.io/attach/cvpr21zhang_acre.pdf)  
Chi Zhang, Baoxiong Jia, Mark Edmonds, Song-Chun Zhu, Yixin Zhu  
*Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021   

Causal induction, *i.e.*, identifying unobservable mechanisms that lead to the observable relations among variables, has played a pivotal role in modern scientific discovery, especially in scenarios with only sparse and limited data. Humans, even young toddlers, can induce causal relationships surprisingly well in various settings despite its notorious difficulty. However, in contrast to the commonplace trait of human cognition is the lack of a diagnostic benchmark to measure causal induction for modern Artificial Intelligence (AI) systems. Therefore, in this work, we introduce the Abstract Causal REasoning (ACRE) dataset for systematic evaluation of current vision systems in causal induction. Motivated by the stream of research on causal discovery in *Blicket* experiments, we query a visual reasoning system with the following four types of questions in either an independent scenario or an interventional scenario: *direct*, *indirect*, *screening-off*, and *backward-blocking*, intentionally going beyond the simple strategy of inducing causal relationships by covariation. By analyzing visual reasoning architectures on this testbed, we notice that pure neural models tend towards an associative strategy under their chance-level performance, whereas neuro-symbolic combinations struggle in backward-blocking reasoning. These deficiencies call for future research in models with a more comprehensive capability of causal induction.

![framework](http://wellyzhang.github.io/img/in-post/ACRE/model.jpeg)

# Dataset

An example ACRE problem is provided below.

![example](http://wellyzhang.github.io/img/in-post/ACRE/example.jpeg)

The dataset formatting document is in ```src/helper/README.md```. To download the dataset, please check [our project page](http://wellyzhang.github.io/project/acre.html#dataset).

# Performance

We show performance of models in the following table. For details, please check our [paper](http://wellyzhang.github.io/attach/cvpr21zhang_prae.pdf).

| Method |      | MXGNet |  LEN   | CNN-MLP |  WReN  | CNN-LSTM | ResNet-MLP | CNN-BERT | NS-RW  | NS-PC  | **NS-Opt** |
|:------:|:----:|:------:|:------:|:-------:|:------:|:--------:|:----------:|:--------:|:------:|:------:|:----------:|
|  IID   | Qry. | 33.01% | 38.08% | 40.86%  | 40.39% |  41.91%  |   42.00%   |  43.56%  | 46.61% | 59.26% | **66.29**% |
|        | Pro. | 1.00%  | 2.05%  |  3.25%  | 2.30%  |  3.60%   |   3.35%    |  3.50%   | 6.45%  | 21.15% | **27.00**% |
| Comp.  | Qry. | 35.56% | 38.45% | 41.97%  | 41.90% |  42.80%  |   42.80%   |  43.79%  | 50.69% | 61.83% | **69.04**% |
|        | Pro. | 1.55%  | 2.10%  |  2.90%  | 2.65%  |  2.80%   |   2.60%    |  2.40%   | 8.10%  | 22.00% | **31.20**% |
|  Sys.  | Qry. | 33.43% | 36.11% | 37.45%  | 39.60% |  37.19%  |   37.71%   |  39.93%  | 42.18% | 62.63% | **67.44**% |
|        | Pro. | 0.60%  | 1.90%  |  2.55%  | 1.90%  |  1.85%   |   1.75%    |  1.90%   | 4.00%  | 29.20% | **29.55**% |


# Dependencies

**Important**
* Python 3.8
* Blender 2.79
* PyTorch
* CUDA and cuDNN expected

See ```requirements.txt``` for a full list of packages required.

# Usage

## Dataset Generation

The dataset generation process consists of two steps: generating dataset configurations and rendering images. A script that combines the two steps is provided in ```dataset_gen.zsh```.

Note that backward-block queries correspond to potential queries in the repo.

### Configuration

Code to generate the dataset configurations resides in the ```src/dataset``` folder. To generate dataset configuration files for a split, run

```
python src/dataset/blicket.py --regime <IID/Comp/SyS> ----output_dataset_dir <directory to save configuration files>
```

### Rendering

Code to render images resides in the ```src/render``` folder. The code is largely adopted from CLEVR. To setup Blender for rendering, 

1. Add the current directory (src/render) to blender's python site-packages 
```
echo $PWD >> <path/to/blender/python/lib/python3.5/site-packages>/acre.pth
```

2. Install pillow for Blender's python

Navigate to ```<path/to/blender/python>``` and run 
```
./bin/python3.5m -m ensurepip
./bin/python3.5m -m pip install pillow
```

To render images, run
```
blender --background -noaudio --python ./src/render/render_images.py -- --use_gpu 1
```

See ```src/render/render_images.py``` for a full list of arguments.

## Benchmarking

Code to benchmark the dataset resides in ```src/baseline``` and ```src/neusym```. 

The ```baseline``` folder contains code for neural networks. Supported models include CNN, LSTM, ResNet, WReN, LEN, MXGNet, and BERT (see ```src/baseline/__init__.py``). To run a model,
```
python src/baseline/main.py --dataset <path/to/ACRE> --model <model name>
```

The ```neusym``` folder contains code for neuro-symbolic models: RWModel, PCModel, NoTearsLinear, and NoTearsMLP (see ```src/neusym/__init__.py```). To run a model,
```
python src/neusym/main.py --split <train/val/test> --config_folder <path/to/ACRE/config> --scene_folder <path/to/predicted/scene/files> --model <model name>
```

We use [Detectron 2](https://github.com/facebookresearch/detectron2) for scene parsing and code for this part is not included in the repo. 

A function to decode masks has been provided in ```helper/mask_decode.py```.

# Citation

If you find the paper and/or the code helpful, please cite us.

```
@inproceedings{zhang2021acre,
    title={Acre: Abstract causal reasoning beyond covariation},
    author={Zhang, Chi and Jia, Baoxiong and Edmonds, Mark and Zhu, Song-Chun and Zhu, Yixin},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2021}
}
```

# Acknowledgement

We'd like to express our gratitude towards all the colleagues and anonymous reviewers for helping us improve the paper. The project is impossible to finish without the following open-source implementation.

* [CLEVR](https://github.com/facebookresearch/clevr-dataset-gen)
* [CLEVR-Ref+](https://github.com/ccvl/clevr-refplus-dataset-gen)
* [NoTears](https://github.com/xunzheng/notears)
* [pcalg](https://github.com/keiichishima/pcalg)
* [RAVEN](https://github.com/WellyZhang/RAVEN)
* [WReN](https://github.com/Fen9/WReN)
* [LEN](https://github.com/zkcys001/distracting_feature)
* [MXGNet](https://github.com/thematrixduo/MXGNet)
* [Detectron2](https://github.com/facebookresearch/detectron2)