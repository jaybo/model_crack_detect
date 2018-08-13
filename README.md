# model_crack_detect

Preforms binary semantic segmentation on tissue samples to detect cracks using VGG16 variants.

``` bash
python train.py -c train
python train.py -c movie
```

Scaled images used in the network are 1920x1829x8, scaled down from the original 3840x3840.  
Augmentation is performed (scaling, rotation, flip).

Training stats on various platforms

| Platform | Model    | configuration                            | disk| batch_size| time_per_batch| time_per_epoch | $/hr | $ to train |
| -------- | -------  | ---------------------------------------- | ----| ----------| --------------| -------------- | ---- | -----------|
|  Azure   | VGG16-32s| Standard ND6s (6 vcpus| 112 GB memory)   |  SSD| 4         | 1             | 21             | 2.50 | $ 4.37     |
|  Azure   | VGG16-8s | Standard ND6s (6 vcpus| 112 GB memory)   |  SSD| 4         | 3             | 62             | 2.50 | $12.91     |
|  Azure   | VGG16-8s | Standard ND6s (6 vcpus| 112 GB memory)   |  SSD| 1         | .8            | 71             | 2.50 | $14.79     |


<a href="http://www.youtube.com/watch?feature=player_embedded&v=YOUTUBE_VIDEO_ID_HERE
" target="_blank"><img src="http://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="480" height="480" border="10" /></a>
