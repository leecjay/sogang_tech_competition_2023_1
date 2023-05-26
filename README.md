## <div align="center">2023-1 서강 융합기술 경진대회 </div>
🚀 yolov5 모델을 활용한 재활용품 분류 SW 및 장치

## <div align="center">Team 서계</div>
🌟 Team Leader 이창재 (서강대학교 기계공학과 19)

🌟 Team member 강정훈 (서강대학교 기계공학과 19)

🌟 Team member 김기훈 (서강대학교 기계공학과 / 컴퓨터공학과 19)

🌟 Team member 이도헌 (서강대학교 기계공학과 19)

## <div align="center">Summary</div>
🚀 Customized yolov5 using Roboflow and Google Colab

- [Our Roboflow Data Set](https://app.roboflow.com/sgme/classify-pet-and-can/4)

- yolov5m 모델을 이용하여 best.pt를 제작


🚀 Take Image using WebCam - terminal (fswebcam)


```bash
import os
picture = "fswebcam --no-banner --set brightness=60% Images/test1.jpg"
os.system(picture)
```


🚀 Analysis the Image - customized yolov5 (detect.py and best.pt)

- Final Image

<p align="center"><img width="800" src="https://file.notion.so/f/s/f78846c6-a9a6-427b-a0f3-4428d04d011c/Untitled.jpeg?id=2e4f2ae3-20c1-45e3-b12a-bc9c71667c68&table=block&spaceId=89f4f652-5ebd-4c52-8a2d-be12a0e49dda&expirationTimestamp=1685187034925&signature=IfXaNab3w6zn-yrGZZgv5DQYqLhXkGZpFzqQxoTMwFQ&downloadName=Untitled.jpeg"></p>
 

- Object 좌표값 데이터를 output.txt에 저장

<p align="center">detect.py</p>

```bash
if len(det):
    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                
    #Customized part : Save det(coordinate info) to output.txt
    for i in range(len(det)):
        print(float(det[i][0]))
        print(float(det[i][1]))
        print(float(det[i][2]))
        print(float(det[i][3]))
        print(float(det[i][5]))
```

<p align="center">Terminal</p>

```bash
python3 /home/sgme/yolov5/detect.py > /home/sgme/yolov5/output.txt --weights /home/sgme/yolov5/best.pt --img 640 --conf 0.4 --source /home/sgme/Images/test1.jpg
```
    
🚀 Make Object Map using 2D Matrix from output.txt Data - Python3 Code(file_read.py) 


🚀 Find the Most Efficient Way - Python3 Code (dijkstra.py)


🚀 Send the Control Order to Arduino Nano by Serial Module - Python3 pyserial


🚀 Recieve the Control Order in the Arduino - Arduino Serial (*.ino)


🚀 Integration Code - Python3 Code (recycle.py)


## <div align="center">Usecase Diagram</div>



## <div align="center">System Context Class Diagram</div>



## <div align="center">Object Structuring</div>

## <div align="center">Collaboration Diagram</div>

## <div align="center">Documentation</div>

## <div align="center">Reference - yolov5</div>

## <div align="center">Documentation</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com/yolov5) for full documentation on training, testing and deployment. See below for quickstart examples.

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

</details>

<details>
<summary>Inference</summary>

YOLOv5 [PyTorch Hub](https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading) inference. [Models](https://github.com/ultralytics/yolov5/tree/master/models) download automatically from the latest
YOLOv5 [release](https://github.com/ultralytics/yolov5/releases).

```python
import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Images
img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

</details>

<details>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources, downloading [models](https://github.com/ultralytics/yolov5/tree/master/models) automatically from
the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.

```bash
python detect.py --weights yolov5s.pt --source 0                               # webcam
                                               img.jpg                         # image
                                               vid.mp4                         # video
                                               screen                          # screenshot
                                               path/                           # directory
                                               list.txt                        # list of images
                                               list.streams                    # list of streams
                                               'path/*.jpg'                    # glob
                                               'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                               'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Training</summary>

The commands below reproduce YOLOv5 [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh)
results. [Models](https://github.com/ultralytics/yolov5/tree/master/models)
and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) download automatically from the latest
YOLOv5 [release](https://github.com/ultralytics/yolov5/releases). Training times for YOLOv5n/s/m/l/x are
1/2/4/6/8 days on a V100 GPU ([Multi-GPU](https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training) times faster). Use the
largest `--batch-size` possible, or pass `--batch-size -1` for
YOLOv5 [AutoBatch](https://github.com/ultralytics/yolov5/pull/5092). Batch sizes shown for V100-16GB.

```bash
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 128
                                                                 yolov5s                    64
                                                                 yolov5m                    40
                                                                 yolov5l                    24
                                                                 yolov5x                    16
```

<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png">

</details>

<details open>
<summary>Tutorials</summary>

- [Train Custom Data](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data) 🚀 RECOMMENDED
- [Tips for Best Training Results](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results) ☘️
- [Multi-GPU Training](https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training)
- [PyTorch Hub](https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading) 🌟 NEW
- [TFLite, ONNX, CoreML, TensorRT Export](https://docs.ultralytics.com/yolov5/tutorials/model_export) 🚀
- [NVIDIA Jetson platform Deployment](https://docs.ultralytics.com/yolov5/tutorials/running_on_jetson_nano) 🌟 NEW
- [Test-Time Augmentation (TTA)](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation)
- [Model Ensembling](https://docs.ultralytics.com/yolov5/tutorials/model_ensembling)
- [Model Pruning/Sparsity](https://docs.ultralytics.com/yolov5/tutorials/model_pruning_and_sparsity)
- [Hyperparameter Evolution](https://docs.ultralytics.com/yolov5/tutorials/hyperparameter_evolution)
- [Transfer Learning with Frozen Layers](https://docs.ultralytics.com/yolov5/tutorials/transfer_learning_with_frozen_layers)
- [Architecture Summary](https://docs.ultralytics.com/yolov5/tutorials/architecture_description) 🌟 NEW
- [Roboflow for Datasets, Labeling, and Active Learning](https://docs.ultralytics.com/yolov5/tutorials/roboflow_datasets_integration)
- [ClearML Logging](https://docs.ultralytics.com/yolov5/tutorials/clearml_logging_integration) 🌟 NEW
- [YOLOv5 with Neural Magic's Deepsparse](https://docs.ultralytics.com/yolov5/tutorials/neural_magic_pruning_quantization) 🌟 NEW
- [Comet Logging](https://docs.ultralytics.com/yolov5/tutorials/comet_logging_integration) 🌟 NEW

</details>

## <div align="center">Why YOLOv5</div>

YOLOv5 has been designed to be super easy to get started and simple to learn. We prioritize real-world results.

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040763-93c22a27-347c-4e3c-847a-8094621d3f4e.png"></p>
<details>
  <summary>YOLOv5-P5 640 Figure</summary>

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/155040757-ce0934a3-06a6-43dc-a979-2edbbd69ea0e.png"></p>
</details>
<details>
  <summary>Figure Notes</summary>

- **COCO AP val** denotes mAP@0.5:0.95 metric measured on the 5000-image [COCO val2017](http://cocodataset.org) dataset over various inference sizes from 256 to 1536.
- **GPU Speed** measures average inference time per image on [COCO val2017](http://cocodataset.org) dataset using a [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) V100 instance at batch-size 32.
- **EfficientDet** data from [google/automl](https://github.com/google/automl) at batch size 8.
- **Reproduce** by `python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n6.pt yolov5s6.pt yolov5m6.pt yolov5l6.pt yolov5x6.pt`

</details>

### Pretrained Checkpoints

| Model                                                                                           | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | Speed<br><sup>CPU b1<br>(ms) | Speed<br><sup>V100 b1<br>(ms) | Speed<br><sup>V100 b32<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>@640 (B) |
| ----------------------------------------------------------------------------------------------- | --------------------- | -------------------- | ----------------- | ---------------------------- | ----------------------------- | ------------------------------ | ------------------ | ---------------------- |
| [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt)              | 640                   | 28.0                 | 45.7              | **45**                       | **6.3**                       | **0.6**                        | **1.9**            | **4.5**                |
| [YOLOv5s](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)              | 640                   | 37.4                 | 56.8              | 98                           | 6.4                           | 0.9                            | 7.2                | 16.5                   |
| [YOLOv5m](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt)              | 640                   | 45.4                 | 64.1              | 224                          | 8.2                           | 1.7                            | 21.2               | 49.0                   |
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt)              | 640                   | 49.0                 | 67.3              | 430                          | 10.1                          | 2.7                            | 46.5               | 109.1                  |
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt)              | 640                   | 50.7                 | 68.9              | 766                          | 12.1                          | 4.8                            | 86.7               | 205.7                  |
|                                                                                                 |                       |                      |                   |                              |                               |                                |                    |                        |
| [YOLOv5n6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n6.pt)            | 1280                  | 36.0                 | 54.4              | 153                          | 8.1                           | 2.1                            | 3.2                | 4.6                    |
| [YOLOv5s6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s6.pt)            | 1280                  | 44.8                 | 63.7              | 385                          | 8.2                           | 3.6                            | 12.6               | 16.8                   |
| [YOLOv5m6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m6.pt)            | 1280                  | 51.3                 | 69.3              | 887                          | 11.1                          | 6.8                            | 35.7               | 50.0                   |
| [YOLOv5l6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l6.pt)            | 1280                  | 53.7                 | 71.3              | 1784                         | 15.8                          | 10.5                           | 76.8               | 111.4                  |
| [YOLOv5x6](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x6.pt)<br>+ [TTA] | 1280<br>1536          | 55.0<br>**55.8**     | 72.7<br>**72.7**  | 3136<br>-                    | 26.2<br>-                     | 19.4<br>-                      | 140.7<br>-         | 209.8<br>-             |

<details>
  <summary>Table Notes</summary>

- All checkpoints are trained to 300 epochs with default settings. Nano and Small models use [hyp.scratch-low.yaml](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-low.yaml) hyps, all others use [hyp.scratch-high.yaml](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-high.yaml).
- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](http://cocodataset.org) dataset.<br>Reproduce by `python val.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`
- **Speed** averaged over COCO val images using a [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) instance. NMS times (~1 ms/img) not included.<br>Reproduce by `python val.py --data coco.yaml --img 640 --task speed --batch 1`
- **TTA** [Test Time Augmentation](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation) includes reflection and scale augmentations.<br>Reproduce by `python val.py --data coco.yaml --img 1536 --iou 0.7 --augment`

</details>


## <div align="center">Contribute</div>

We love your input! We want to make contributing to YOLOv5 as easy and transparent as possible. Please see our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) to get started, and fill out the [YOLOv5 Survey](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey) to send us feedback on your experiences. Thank you to all our contributors!

<!-- SVG image from https://opencollective.com/ultralytics/contributors.svg?width=990 -->

<a href="https://github.com/ultralytics/yolov5/graphs/contributors">
<img src="https://github.com/ultralytics/assets/raw/main/im/image-contributors.png" /></a>

## <div align="center">License</div>

YOLOv5 is available under two different licenses:

- **AGPL-3.0 License**: See [LICENSE](https://github.com/ultralytics/yolov5/blob/master/LICENSE) file for details.
- **Enterprise License**: Provides greater flexibility for commercial product development without the open-source requirements of AGPL-3.0. Typical use cases are embedding Ultralytics software and AI models in commercial products and applications. Request an Enterprise License at [Ultralytics Licensing](https://ultralytics.com/license).
  
  
  
  

