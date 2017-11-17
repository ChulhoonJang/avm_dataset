# AVM (Around View Monitoring) System Datasets for Auto Parking

## Abstract

We present the AVM System Datasets for auto parking. The datasets consists of two different categories. One aims for training of semantic segmentation to understand surrounding environments by using only AVM images. The other aims for performance evaluation of parking space detection. We hope that through these datasets, many researchers will suggest creative algorithms and improve recognition performance.



## Description of *SS*(Semantic Segmentation) dataset

This dataset contains 6414 camera images at a resolution of 320 x 160 pixels. There are four categories: free space, marker, vehicle, and other objects. For each image, a corresponding ground truth image is composed of four color annotations to distinguish different classes.

dataset name: *AVM6414* ([download](https://drive.google.com/drive/folders/0BylKEIPbxlTtbUtHbjdhV2JLSGM?usp=sharing))

| Category  | Frames   |
| --------- | -------- |
| Training  | 3849     |
| Valid     | 962      |
| Test      | 1603     |
| **Total** | **6414** |

* class 0: Free space *- RGB color [0, 0, 255]*
* class 1: Marker *- RGB color [255,255,255]*
* class 2: Vehicle *- RGB color [255,0,0]*
* class 3: Other objects (curb, pillar, wall, and so on) *- RGB color [0,255,0]*
* Negligible area: Ego vehicle *- RGB color [0,0,0]*

![image](ss_dataset_example/image.jpg) ![gt](ss_dataset_example/gt.png)

The SS dataset contains various samples from outdoor and indoor parking lots. In particular, the indoor samples are quite difficult to recognize because reflected lights look similar with slot markers and they might degrade slot marker detection.

![samples](ss_dataset_example/avm_image_samples.png)

​                                                       (a) outdoor-day, (b) outdoor-rainy, (c) indoor

## Description of *PS*(Parking Space) dataset

coming soon



