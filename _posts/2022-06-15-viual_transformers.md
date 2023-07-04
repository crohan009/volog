---
layout: post
comments: false
title: "Vision Transformers"
subtitle: "Transformers in Computer Vision"
date: 2022-06-15 00:01:01
tags:
---

> This article dives in the world Transformer-based models used in computer vision (CV) by analyzing the [Vision Transformer](https://arxiv.org/abs/2010.11929) and [Detection Transformer](https://arxiv.org/abs/2005.12872) models and explores their adoption for end-to-end CV training, eliminating intermediate hand-crafted components; and their utilization of positional encodings to capture the spatial layout of images.

<!--more-->

---
<h3> Contents </h3>

{: class="table-of-content"}
* TOC
{:toc}

---

## **DETR & ViT: The Intersecting Pathways**

The world of computer vision has seen revolutionary advancements, with Transformers as a notable pillar of change. Originally designed for sequence transduction tasks in Natural Language Processing (NLP), the Transformer architecture has proven to be versatile, finding purpose in vision tasks in DETR (Detection Transformer) and ViT (Visual Transformer) papers. 

Both these papers leveraged the Transformers' unique ability to model long-range dependencies, effectively handling spatial relations between different parts of an image. A standout feature of both DETR and ViT is their adoption of end-to-end training, eschewing intermediate hand-crafted components. DETR discards the need for Non-Maximum Suppression (NMS) and anchor generation. On the other hand, ViT directly trains on image pixels, effectively discarding the need for convolutional layers.

Another shared characteristic between DETR and ViT is the utilization of positional encodings to incorporate the spatial layout of the image into the model. This is especially vital in computer vision tasks, where order is not inherently meaningful as it is in NLP.

---

## 1. DETR

The DETR (DEtection TRansformer) model is a novel object detection model that uses a transformer architecture (both the **transformer encoder** and the **transformer decoder**). 

![DeTr]({{ '/assets/images/vision_transformers/detr_01.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 1. Detection Transformer (DeTr).*

Here's a breakdown of its main components:

#### 1. Backbone

The backbone of the DETR model is a convolutional neural network (CNN) that is used to extract features from the input image. This is typically a pre-trained model, such as ResNet, that has been trained on a large image classification task like ImageNet. The backbone takes an image as input and outputs a set of feature maps that represent the content of the image at different levels of abstraction. These feature maps are then used as input to the transformer encoder.

#### 2. Encoder

The encoder is a transformer model that takes the feature maps from the backbone as input and generates a set of encoded feature maps. The transformer model is composed of a stack of identical layers, each with two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. 

The self-attention mechanism allows the model to weigh the importance of different parts of the image when generating the encoded feature maps. This is done by computing a weighted sum of the input feature maps, where the weights are determined by the attention scores. 

The position-wise fully connected feed-forward network is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

#### 3. Decoder

The decoder is also a transformer model, but it has an additional set of inputs called "object queries". These queries are learned during training and are used to interact with the encoded feature maps to generate the final object detections.

Each object query is associated with a potential object in the image, and through the interaction with the encoded feature maps, it learns to represent the properties of that object, such as its class and bounding box coordinates.

The decoder is composed of a stack of identical layers, each with three sub-layers: a multi-head self-attention mechanism, a multi-head attention mechanism over the encoder output, and a position-wise fully connected feed-forward network.

#### 4. Prediction Heads

The prediction heads are the final part of the DETR model. They take the output of the decoder and generate the final object detections. There are two prediction heads: one for predicting the class of each object, and one for predicting the bounding box coordinates.

The class prediction head is a simple linear layer followed by a softmax activation function, which outputs a probability distribution over the possible object classes.

The bounding box prediction head is a linear layer that outputs four values representing the coordinates of the bounding box.

### The DETR module

- **Backbone**: The backbone is a convolutional neural network (CNN) that is used to extract features from the input image. In this case, the backbone is a ResNet-50 model with the last two layers (average pooling and fully connected layer) removed. The backbone takes an image as input and outputs a set of feature maps.

- **Convolutional Layer**: The convolutional layer (`self.conv`) is used to transform the feature maps from the backbone into a new set of feature maps with a specified number of channels (`hidden_dim`). This is done using a 1x1 convolution, which is equivalent to applying a linear transformation to each pixel independently.

- **Transformer**: The transformer (`self.transformer`) is a standard transformer model with a specified number of heads (`nheads`) and a specified number of layers in the encoder and decoder (`num_encoder_layers` and `num_decoder_layers`). The transformer takes the feature maps from the convolutional layer and the positional encodings as input and outputs a set of feature maps.

- **Prediction Heads**: The prediction heads are two linear layers that are used to predict the class and bounding box coordinates for each object query. The class prediction head (`self.linear_class`) outputs a vector of size `num_classes + 1` for each query, representing the probabilities of each class and an additional "no object" class. The bounding box prediction head (`self.linear_bbox`) outputs a vector of size 4 for each query, representing the coordinates of the bounding box.

- **Query Positional Encodings (`self.query_pos`)**: These are learned positional encodings for the object queries. Its dimension `[100, 256]` implies there are 100 object queries, each with a 256-dimensional positional encoding. Each object query is associated with a potential object in the image, and its positional encoding is used to provide positional information to the transformer.

- **Row and Column Positional Encodings (`self.row_embed` and `self.col_embed`)**: These are learned positional encodings for the rows and columns of the image. Its dimensions `[50, 128]` means there are 50 positional encodings for the rows and 50 for the columns, each with a 128-dimensional encoding. These positional encodings are used to provide positional information to the transformer about the location of each pixel in the image.


```python
import torch
from torch import nn
from torchvision.models import resnet50

class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers):
        super().__init__()
        # We take only convolutional layers from ResNet-50 model
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone(inputs)
        h = self.conv(x)
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        h = self.transformer(pos + h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1))
        return self.linear_class(h), self.linear_bbox(h).sigmoid()

detr = DETR(num_classes=91, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6)
detr.eval()
inputs = torch.randn(1, 3, 800, 1200)
logits, bboxes = detr(inputs)
```
Source: [End-to-End Object Detection with Transformers, Pg: 26](https://arxiv.org/abs/2005.12872)

### Forward Pass

- **Backbone and Convolutional Layer**: The input images are first passed through the backbone and the convolutional layer. The backbone extracts a set of feature maps from the images (`x` with shape [1, 2048, 25, 38], where 1 is the batch size, 2048 is the number of channels, and 25x38 is the spatial resolution), and the convolutional layer transforms these feature maps into a new set of feature maps with a specified number of channels (`h` with shape [1, 256, 25, 38], where 256 is the `hidden_dim`).

- **Positional Encodings**: The positional encodings for the rows and columns of the image are concatenated together to form a 2D positional encoding for each pixel in the image. These 2D positional encodings (`pos` with shape [950, 1, 256], where 950 is the total number of pixels in the image, 1 is the batch size, and 256 is the `hidden_dim`) are then flattened and unsqueezed to match the shape of the feature maps.

- **Transformer**: The feature maps and the positional encodings are added together and passed through the transformer. The transformer takes the feature maps and the positional encodings as input and outputs a set of feature maps (`h` with shape [100, 1, 256], where 100 is the number of object queries, 1 is the batch size, and 256 is the `hidden_dim`). The object queries are also passed to the transformer as an additional input.

- **Prediction Heads**: The output of the transformer is passed through the prediction heads to generate the final predictions. The class prediction head outputs a vector of class probabilities for each object query, and the bounding box prediction head outputs a vector of bounding box coordinates for each object query. The bounding box predictions are passed through a sigmoid function to constrain them to the range [0, 1].



### Bipartite Matching Loss

#### Step #1 - The Matching Cost

The DETR (DEtection TRansformer) model uses the `HungarianMatcher` class, which is used to match the predicted bounding boxes and classes (from the DETR model) with the ground truth bounding boxes and classes. The matching is done in such a way as to minimize the total cost, which is a combination of the classification cost, the bounding box cost, and the Generalized Intersection over Union (GIoU) cost.

##### Cost Matrix "C" Creation

The cost matrix "`C`" is a combination of three different costs: *classification* cost, *bounding box* cost, and *GIoU* cost. Each of these costs are computed separately and then combined to form the final cost matrix "`C`".

1. **Classification Cost (`cost_class`)**: This cost is computed as the *negative of the softmax probability* of the predicted class for each target class. In other words, it's `1` minus the probability that the model assigns to the correct class. This cost is high when the model is uncertain about the correct class.

2. **Bounding Box Cost (`cost_bbox`)**: This cost is computed as the *L1 distance* (sum of absolute differences) between the predicted bounding box coordinates and the target bounding box coordinates. This cost is high when the predicted bounding box is far from the target bounding box.

3. **GIoU Cost (`cost_giou`)**: This cost is computed as the *negative of the Generalized Intersection over Union (`GIoU`)* between the predicted bounding box and the target bounding box. GIoU is a measure of overlap between two bounding boxes, and it ranges from -1 to 1. A GIoU of 1 means the bounding boxes are a perfect match, a GIoU of 0 means they don't overlap at all, and a GIoU of -1 means the bounding boxes are the worst possible match. By taking the negative of the GIoU, we get a cost that is high when the bounding boxes are a poor match.

The final cost matrix "`C`" is a weighted sum of these three costs, where the weights are the parameters `cost_class`, `cost_bbox`, and `cost_giou` that are passed to the `HungarianMatcher` constructor. The cost matrix "`C`" is then reshaped and moved to the CPU for the subsequent matching process.


![bipartite_matching_cost]({{ '/assets/images/vision_transformers/bipartite_matching_cost.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 4. Bipartite Matching Loss [[Source](https://github.com/facebookresearch/detr/blob/main/models/matcher.py)].*

#### Step #2 - The Hungarian Loss 

The second step of the matching process uses the 'Hungarian algorithm' (also known as the Kuhn-Munkres algorithm) to find the optimal assignment of predictions to targets that minimizes the total cost. This is done using the `linear_sum_assignment` function from the `scipy.optimize` module. The result is a list of tuples, where each tuple contains the indices of the selected predictions and the corresponding selected targets.

The term "Hungarian" comes from the Hungarian algorithm, also known as the Kuhn-Munkres algorithm or the assignment problem, is a combinatorial optimization algorithm which is used to find a perfect matching in a weighted bipartite graph that minimizes the sum of weights of the edges. It solves the assignment problem in polynomial time.

##### Sizes and Indices Calculation

1. **Indices Calculation (`indices`)**: This line of code performs the Hungarian algorithm on each cost matrix in the batch. The `C.split(sizes, -1)` expression splits the cost matrix `C` into a list of cost matrices, one for each target in the batch. The `linear_sum_assignment(c[i])` expression performs the Hungarian algorithm on the `i`-th cost matrix, and the list comprehension `[linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]` performs this operation for each cost matrix in the batch. The result is a list of tuples, where each tuple contains the indices of the selected predictions and the corresponding selected targets for a single target.

2. **Return Statement**: This line of code converts the indices to PyTorch tensors and returning them. The `(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))` expression converts the indices `i` and `j` to PyTorch tensors of type `int64`, and the list comprehension `[(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]` performs this operation for each pair of indices in the `indices` list. The result is a list of tuples, where each tuple contains the indices of the selected predictions and the corresponding selected targets for a single target, in the form of PyTorch tensors.


![hungarian_loss]({{ '/assets/images/vision_transformers/hungarian_loss.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 4. Hungarian Loss [[Source](https://github.com/facebookresearch/detr/blob/main/models/matcher.py)]*


This returned list of tuples is the final output of the `HungarianMatcher`'s forward method. Each tuple in the list corresponds to a single image in the batch, and contains two tensors of `indices`: the first tensor contains the indices of the selected predictions, and the second tensor contains the indices of the corresponding selected targets. These `indices` can be used to gather the selected predictions and targets from the outputs and targets variables, respectively.


## 2. ViT

![ViT_model_overview]({{ '/assets/images/vision_transformers/ViT_model_overview.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 4. Vision Transformer [[Source](https://arxiv.org/abs/2010.11929)]*

### 2.1 Introduction

The Vision Transformer (ViT) is a model that applies the **transformer encoder architecture** (as seen on the right side of the figure), which has been highly successful in discriminative NLP tasks, directly to image classification tasks. Unlike previous approaches that combine convolutional neural networks (CNNs) with self-attention mechanisms, ViT uses a pure transformer (encoder) architecture and treats an image as a sequence of patches, similar to how a transformer model in NLP treats a sentence as a sequence of words.

### 2.2 Model Architecture

#### 2.2.1 Patch Embedding

The first step in the ViT model is to divide the input image into a fixed number of patches, each of size 16x16 pixels. These patches are then linearly transformed (or "embedded") into a sequence of vectors, which serve as the input to the transformer model. This is similar to how words in a sentence are embedded into vectors in NLP.

#### 2.2.2 Transformer Encoder

The sequence of patch embeddings is then passed through a standard transformer encoder, which consists of a stack of transformer layers. Each layer in the transformer encoder contains a multi-head self-attention mechanism and a position-wise feed-forward network.

#### 2.2.3 Classification Head

The output of the transformer encoder is a sequence of vectors, one for each input patch. The vector corresponding to the first patch (or the "class" token in NLP terminology) is used as the image representation and is passed through a classification head to predict the image label.

### 2.3 Training and Evaluation

ViT models are typically pre-trained on a large dataset, such as ImageNet, and then fine-tuned on a smaller target dataset. Despite their lack of inductive biases that are inherent to CNNs, such as local receptive fields and translation equivariance, ViT models have achieved competitive, and in some cases, state-of-the-art performance on various image classification benchmarks when trained at scale.

### 2.4 Key Findings and Contributions

The key contributions of the ViT model are:

- Demonstrating that the transformer architecture can be applied directly to image classification tasks, without the need for CNNs or other image-specific components.
- Showing that despite their lack of inductive biases, transformer models can achieve competitive performance on image classification tasks when trained at scale.
- Proposing a new way of treating images as sequences of patches, which allows for the direct application of transformer models to images.

### 2.5 Why Use Transformers for Image Classification?

While Convolutional Neural Networks (CNNs) have been the go-to models for image classification tasks, the Vision Transformer (ViT) takes a different approach by applying the transformer architecture directly to images. Here are the key reasons for this choice:

- **Modeling Long-Range Dependencies**: Transformers are capable of modeling long-range, global dependencies within the data. In the context of images, this means that a transformer can capture relationships between pixels that are far apart, something that CNNs, with their focus on local, nearby pixels, struggle with.

- **Scalability**: Transformers are more flexible and scalable than CNNs. They can be easily scaled up by increasing the number of layers or the model size, which often leads to better performance. This is not always the case with CNNs, which can suffer from diminishing returns when scaled up.

- **Potential Representational Power**: The success of transformers in NLP has shown that they are capable of learning useful representations from data, even without strong inductive biases. This suggests that transformers could also learn to recognize useful patterns in images, given enough data and computational resources.

In summary, while CNNs are more efficient parametrically, transformers offer advantages in terms of modeling long-range dependencies, scalability, and potential representational power. These advantages make transformers an interesting and promising alternative to CNNs for image classification tasks.

---

## **DETR vs ViT: Notable Differences**

While DETR and ViT both harness the power of Transformers, they apply it in different ways to address distinct challenges in computer vision.

1. **Tasks:** DETR was designed with object detection and panoptic segmentation tasks in mind, while ViT primarily addresses image classification tasks.

2. **Input Representation:** DETR maintains the original spatial resolution of the image, flattening it into a sequence of image patches for the transformer encoder. In contrast, ViT divides the image into a fixed number of patches, each of which is represented as a token for the Transformer.

3. **Model Components:** DETR features an object detection head on top of the Transformer and uses a set prediction approach with bipartite matching loss. Additionally, it incorporates an encoder-decoder Transformer structure for reasoning about object relations. ViT employs a pure Transformer encoder and adds a simple classification head on top of it.

4. **Training Strategies:** DETR employs a matching cost and a Hungarian algorithm for pairing predicted and ground-truth objects, a considerable departure from traditional methods. ViT uses standard supervised learning with a cross-entropy loss for training.

5. **Performance on Small Objects:** DETR has shown difficulty in dealing with small objects, a challenge that is not explicitly mentioned in the ViT paper.

6. **Architectural Variations:** DETR utilises a more complex architecture with multiple encoder and decoder layers. ViT, in contrast, follows a more straightforward, streamlined architecture.

---
