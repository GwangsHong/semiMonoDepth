# semiMonoDepth
Tensorflow implementation of semi-supervised monocular depth estimation

## Requirements
```
Ubuntu 14.04
Tensorflow 1.7
python 3.5
opencv3 3.1.0
```
You can train on multiple GPUs by setting them with the --num_gpus flag, make sure your batch_size is divisible by num_gpus.

## Data
This model requires rectified stereo pairs for training.
There are two main datasets available:
### [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php)
```
.
├── ./kitti_dataset
│   ├── ./kitti_dataset/2011_09_26
│   │   ├── ./kitti_dataset/2011_09_26/2011_09_26_drive_0001_sync
│   │   │   ├── ./kitti_dataset/2011_09_26/2011_09_26_drive_0001_sync/image_00
│   │   │   │   ├── ./kitti_dataset/2011_09_26/2011_09_26_drive_0001_sync/image_00/data
│   │   │   │   │   ├── ./kitti_dataset/2011_09_26/2011_09_26_drive_0001_sync/image_00/data/0000000000.png
│   │   │   │   │   └── ...

```
You can download the entire raw dataset by running:
```
wget -i utils/kitti_archives_to_download.txt -P ~/kitti_dataset/
```

Google drive

[https://drive.google.com/open?id=1AKxvM3O5aH3GHtbme_8CULGG-OJlht6s](https://drive.google.com/open?id=1AKxvM3O5aH3GHtbme_8CULGG-OJlht6s)

### [Scene flow dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
```
├── ./sceneflow_dataset
│   └── ./sceneflow_dataset/flyingthings3d
│       ├── ./sceneflow_dataset/flyingthings3d/disparity
│       │   ├── ./sceneflow_dataset/flyingthings3d/disparity/TRAIN
│       │   │   ├── ./sceneflow_dataset/flyingthings3d/disparity/TRAIN/A
│       │   │   │   ├── ./sceneflow_dataset/flyingthings3d/disparity/TRAIN/A/0000
│       │   │   │   │   ├── ./sceneflow_dataset/flyingthings3d/disparity/TRAIN/A/0000/left
│       │   │   │   │   │   ├── ./sceneflow_dataset/flyingthings3d/disparity/TRAIN/A/0000/left/0006.pfm
│       │   │   │   │   │   └── ...
```
We also used synthesis dataset for semi-supervised learning. 
```
wget -i utils/scenflow_archives_to_download.txt -P ~/sceneflow_dataset/
```
Google drive

[https://drive.google.com/open?id=1yUOaVXFzCstIRamxUvMMIemqfN7forkk](https://drive.google.com/open?id=1yUOaVXFzCstIRamxUvMMIemqfN7forkk)

## Traning
The model has three steps: learnig for generator using semi-supervised loss, learning for discriminator, learning for generator. 
* --train_step 0: learnig for generator using semi-supervised loss
* --train_step 1: learning for discriminator: 
* --train_step 2: learning for generator 

train_step 1 and 2 are required to load the checkpoint, you can be doen with --checkpoint_path
```
python semiMonoDepth_main.py --mode train --labeled_data_path ~/sceneflow_dataset/ --unlabeled_data_path ~/kitti_dataset/ --labeled_filenames_file utils/filenames/flyingthings3d_train_shuffle_files.txt --unlabeled_filenames_file utils/filenames/kitti_train_files.txt   --train_step 0  --log_directory ~/tmp/
```

## Testing
To test change the --mode flag to test, the network will output the disparities in the model folder or in any other folder you specify wiht --output_directory.
You will also need to load the checkpoint you want to test on, this can be done with --checkpoint_path:

### Evaluation on KITTI
To evaluate run:
```
python utils/evaluate_kitti.py --split kitti --predicted_disp_path ~/tmp/my_model/disparities.npy \
--gt_path ~/data/KITTI/
```
