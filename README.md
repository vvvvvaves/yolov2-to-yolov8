# From YOLOv2 to YOLOv8: a Seven-Year Leap in Object Detection
The following repository contains the code from my thesis. The thesis explores the advancements made in object detection between 2016 and 2023 by comparing the architectures of YOLOv2 and YOLOv8. 

The experiment is divided into two parts. The first part compares the backbones of YOLOv2 and YOLOv8. The second part compares YOLOv2 and YOLOv8 detection models. The comparison is done by taking a baseline model and building it up until it matches the final model. The build-up is divided into steps, with each step adding a single or multiple design elements.

Imagenette2 is used as a dataset for classification training.

## Backbone models compared in the study:

### Baseline: Darknet-19
It is implemented in accordance with the YOLOv2 paper.

### Experimental model: Version 1
This model can also be called “bare YOLOv8.” It has the same blocks as YOLOv8-cls, yet, they are stripped of constructs which were not implemented in YOLOv2. It is half the size of Darknet-19, which makes the comparison between them impossible. But it is an architecture with the same set of techniques and, hence, serves as a reference point for further steps in the experiment. In summary, the model:
- Does not have residual connections
- Uses the classification head of Darknet-19
- C2f blocks do not implement CSP strategy (they are just a sequence of consecutive convolutions without the concatenation of hidden outputs).

### Experimental model: Version 2
The second model builds off of the first one. It is the same, except it has residual connections implemented in every bottleneck block.

### Experimental model: Version 3
The third model is the same as the second one, except it implements Cross Stage Partial strategy. It does not concatenate hidden outputs.

### Experimental model: Version 4
The fourth model is the same as the third one, except it concatenates hidden outputs of the bottleneck blocks.

### Final model: YOLOv8s-cls (Version 5)

The fifth model is the same as the fourth one, except it uses YOLOv8 classification head instead of YOLOv2 classification head. While Darknet-19 is fully convolutional, YOLOv8 classifier uses a fully connected layer to make predictions.

The fifth model fully matches the YOLOv8 classifier. It is the end model of backbone experiments.

|                                 | Darknet-19 | Version 1 | Version 2 | Version 3 | Version 4 | YOLOv8s-cls |
| :---------------------------:   | :--------: | :-------: |  :------: | :-------: | :-------: | :-------:   |
| Residual connections            |            |           | ✓        | ✓         | ✓         | ✓           |
| CSP                             |            |           |           | ✓        | ✓         | ✓           |
| Concatenation of hidden outputs |            |           |           |          | ✓         | ✓           |
| YOLOv8 classification head      |            |           |           |           |           | ✓           |

## Backbone experiments: results
|                                 | Darknet-19 | Version 1 | Version 2 | Version 3 | Version 4 | YOLOv8s-cls |
| :---------------------------:   | :--------: | :-------: |  :------: | :-------: | :-------: | :-------:   |
| Inference, ms [640]           | 379.794      |  216.398  | 211.871   | 114.806   | 120.620   | 117.802     |
| Inference, ms [224]           | 70.62        | 51.919    | 54.367    | 40.979    | 43.748    | 25.422      |
| Parameters, M                 | 20.8         | 10.5      | 10.5      | 4.7       | 4.9       | __6.4__     |
| GFLOPs (torch) [640]          | 45.57        | 28.69     | 28.69     | 11.70     | 12.33     | 12.70       |
| GFLOPs (thop) [640]           | 45.86        | 28.86     | 28.86     | 11.84     | 12.47     | __13.61__   |
| GFLOPs (torch) [224]          | __5.58__     | 3.51      | 3.51      | 1.43      | 1.51      | 1.56        |
| GFLOPs (thop) [224]           | 5.62         | 3.54      | 3.54      | 1.45      | 1.53      | 1.67        |
| Peak Memory Usage, MB [640]   | 100.0        | 31.4      | 31.4      | 34.4      | 40.6      | 40.6        |
| Peak Memory Usage, MB [224]   | 18.5         | 9.5       | 9.5       | 5.0       | 5.0       | 5.0         |
| Validation accuracy, % [224] [20 epochs]                    | 76.74%       | 67.82%    | 73.99%    | 75.69%    | 74.93%    | 76.54%      |

__Bold__ indicates measures that align with those presented by the authors of original implementations.




















