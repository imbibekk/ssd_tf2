# SSD-tf2.0
tf2.0 Implementation of SSD

*This repo is inspired by [SSD Pytorch](https://github.com/lufficc/SSD) and can be seen as its porting in tf2.0*

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
conda create -n ssd_tf2 python=3.6
source activate ssd_tf2
pip install -r requirements.txt
```

## Prepare dataset
You need VOC 2007 and VOC 2012 data. If you don't have already, you can download it by
```
$ sh download_voc.sh
```
After downloading and unzipping, the data directory should look like this:
```
data
  +- pascal_voc
    +- VOCdevkit
      +- VOC2007
      +- VOC2012
```

## Training
Training can be done by using the config file in `configs` folder.
```
python main.py --config configs/vgg_ssd300_voc0712.yaml
```
*`log.txt` file is attached for your reference

## Evaluate
For evaluating the trained model
```
python main.py --config configs/vgg_ssd300_voc0712.yaml --test True CKPT './LOGS/model_weights.h5'
```
