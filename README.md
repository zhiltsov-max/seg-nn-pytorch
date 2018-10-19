# Implementation of semantic segmentation for images in Pytorch

Template project for experiments with semantic segmentation task.

# Prerequisites

- Python 2.6+ / 3.5+
- PyTorch 0.4.0+
- Pillow
- NumPy

# How to run

## Data setup

Put dataset somewhere and create directory structure as follows:
```
dataset/
dataset/list/train.txt
dataset/list/val.txt
dataset/list/test.txt
dataset/images/
dataset/gt/
```

Lists contain lines in following format:

```
path/to/image path/to/ground/truth
```

Pathes are expected to be relative to `dataset` folder, which is described above.

For each dataset an adapter class should be created in `nn_compression/datasets` directory. 
Currently there are `CamVid12` and `CamVid32` adapters are implemented.

## Launch

To run training or testing there is the `main.py` script. It has many execution options, 
which can be listed by running `python main.py --help`.

To run experiment add parent directory of `nn_compression` to `PYTHONPATH` environment variable
and launch `main.py` script.

``` bash
# Linux
# git clone https://github.com/zhiltsov-max/seg-nn-pytorch && cd seg-nn-pytorch
export PYTHONPATH=$PWD:$PYTHONPATH
python src/main.py --dataset CamVid32 --data_dir path/to/camvid --batch_size 4
```

``` powershell
# Windows-PowerShell (consider Anaconda installation)
# git clone https://github.com/zhiltsov-max/seg-nn-pytorch && cd seg-nn-pytorch
$env:PYTHONPATH="<current dir>"
python src/main.py --dataset CamVid32 --data_dir path/to/camvid --batch_size 4
```
