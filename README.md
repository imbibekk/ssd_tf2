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
*`log.txt` file is attached for your reference*

## Evaluate
For evaluating the trained model. Model weights can be downloaded via this [link](https://www.dropbox.com/s/6femfgitaguktqq/model_weights.h5?dl=0)

```
python main.py --config configs/vgg_ssd300_voc0712.yaml --test True CKPT model_weights.h5
```
```
AP_aeroplane : 0.835755892121836
AP_bicycle : 0.8477451970587925
AP_bird : 0.7581270062896923
AP_boat : 0.7155429490946621
AP_bottle : 0.5119526888744691
AP_bus : 0.8570726602180382
AP_car : 0.8581651770387565
AP_cat : 0.8876963997340174
AP_chair : 0.6203951857593561
AP_cow : 0.8180239944786718
AP_diningtable : 0.7644197478257684
AP_dog : 0.847685403061531
AP_horse : 0.8666407676532754
AP_motorbike : 0.8243487261074417
AP_person : 0.7925955925050024
AP_pottedplant : 0.5233091994468501
AP_sheep : 0.7632801493862849
AP_sofa : 0.8068765880115991
AP_train : 0.8556612806527891
AP_tvmonitor : 0.7674626899773555
mAP : 0.7761378647648095
```

