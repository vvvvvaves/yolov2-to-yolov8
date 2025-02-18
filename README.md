# From YOLOv2 to YOLOv8: a Seven-Year Leap in Object Detection
The following repository contains the code from my thesis. The thesis explores the advancements made in object detection between 2016 and 2023 by comparing the architectures of YOLOv2 and YOLOv8. 

The experiment is divided into two parts. The first part compares the backbones of YOLOv2 and YOLOv8. The second part compares YOLOv2 and YOLOv8 detection models. The comparison is done by taking a baseline model and building it up until it matches the final model. The build-up is divided into steps, with each step adding a single or multiple design elements.

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

## Backbone experiments: results

| | Baseline (Darknet-19) | Version 1 | Version 2 | Version 3 | Version 4 | Version-5 (YOLOv8s-cls) |
| :----:   | :----: | :----: |  :----: | :----: | :----: | :----:   |
| Residual connections |    | ✓   | 301   | 283   | 283   |
| CSP |    | ✓   | 301   | 283   | 283   |
| Concatenation of hidden outputs |    | ✓   | 301   | 283   | 283   |
| YOLOv8 classification head |    | ✓   | 301   | 283   | 283   |





