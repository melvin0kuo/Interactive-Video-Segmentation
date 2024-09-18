# 互動式影像分割專題報告 / Interactive Video Segmentation Project Report
## TKU AI 4 Thesis
[中文版本](#中文) | [English Version](#english)

---

## 中文

### 專案簡介

本研究旨在結合語言輸入，實現即時互動的影像分割，以達到即時互動的效果。此專案主要針對在資源有限的邊緣設備上進行優化，使用輕量化語言模型來達成即時影像分割任務。

### 組員
- 蘇家樂 (410000159)
- 郭傲羲 (410000175)

### 指導老師
李宜勳教授

### 機構
淡江大學人工智慧學系

### 目標
- 減小語言模型大小，同時保持準確性。
- 使用模型輕量化技術以實現高效部署。
- 使模型能夠在邊緣設備上進行即時影像處理。
- 在實際邊緣設備（如 Nvidia Jetson ORIN-NANO）上成功部署。

### 方法

#### 模型架構
本專案使用多模型特徵融合方法，結合 ResNet101 影像特徵提取與輕量化 BERT 語言特徵，並透過 Atrous 空洞卷積進行多尺度特徵提取，使用互注意力機制增強特徵融合。

![模型架構圖](path_to_model_image)

#### 模型輕量化
將 PyTorch 模型轉換為 ONNX 格式，並使用 TensorRT 進行優化，減少推理時間與資源佔用，使模型適用於邊緣設備。

![模型輕量化流程](path_to_conversion_image)

### 實驗設置
#### 訓練設備
- 設備 1: 
  - CPU：i7-9700
  - RAM：32GB DDR4
  - GPU：RTX 3090 24GB
- 設備 2: 
  - CPU：i5-14600k
  - RAM：16GB DDR5
  - GPU：RTX 3090 24GB

#### 推理設備
- Jetson Orin Nano (8GB RAM)
- Jetpack 版本：Jetpack 6.0
- TensorRT 版本：8.6.2.3-1
- CUDA：12.2

### 結果

#### 模型性能
輕量化的 BERT 模型顯著提升了推理速度，同時保持高準確性。經 TensorRT 優化後，模型在邊緣設備上的性能進一步提升。

| 模型               | 平均 IoU | 總體 IoU | 參數量         |
|--------------------|----------|----------|----------------|
| RefVOS             | 42       | 41       | 160,000,000    |
| RefVOS + co-attention | 41    | 41       | 71,000,000     |
| Frozen Albert       | 60       | 57       | 58,880,000     |

#### TensorRT 優化影響
經 TensorRT 優化後，推理時間大幅縮短，內存使用量也顯著降低。

| 模型              | 推理時間 (秒) | 內存使用量 (MB) |
|-------------------|--------------|----------------|
| BERT（原始）      | 0.24979      | 417            |
| BERT（TensorRT）  | 0.01575      | 54             |
| Albert（原始）    | 0.25598      | 44             |
| Albert（TensorRT）| 0.01682      | 54             |

### 未來計劃
- 提高 TensorRT 模型的準確性
- 增加語音輸入功能
- 建立專屬訓練數據集

### 結論
本專案成功將大型 BERT 模型進行輕量化並部署至邊緣設備上，實現了高效的即時互動影像分割。未來我們將繼續優化模型，增加功能，以期在更多應用場景中發揮作用。

---

## English

### Project Overview

This study aims to implement real-time interactive image segmentation by combining language input, achieving an interactive effect. The project focuses on optimizing the system for edge devices with limited resources, using a lightweight language model to achieve real-time image segmentation tasks.

### Team Members
- Su Jia-Le (410000159)
- Guo Ao-Xi (410000175)

### Supervisor
Professor Lee Yi-Hsün

### Institution
Tamkang University, Department of Artificial Intelligence

### Objectives
- Reduce the size of the language model while maintaining accuracy.
- Utilize model lightweighting techniques for efficient deployment.
- Enable real-time image processing on edge devices.
- Deploy the model successfully on real edge devices, such as Nvidia Jetson ORIN-NANO.

### Methods

#### Model Architecture
This project uses a multi-model feature fusion approach, combining ResNet101 image feature extraction with lightweight BERT language features. Atrous convolution is used for multi-scale feature extraction, and co-attention mechanisms are introduced to enhance feature fusion.

![Model Architecture](path_to_model_image)

#### Model Lightweighting
The PyTorch model is converted to ONNX format and optimized using TensorRT, reducing inference time and resource consumption, making the model suitable for edge devices.

![Model Lightweighting Process](path_to_conversion_image)

### Experiment Setup
#### Training Devices
- Device 1: 
  - CPU: i7-9700
  - RAM: 32GB DDR4
  - GPU: RTX 3090 24GB
- Device 2: 
  - CPU: i5-14600k
  - RAM: 16GB DDR5
  - GPU: RTX 3090 24GB

#### Inference Devices
- Jetson Orin Nano (8GB RAM)
- Jetpack Version: Jetpack 6.0
- TensorRT Version: 8.6.2.3-1
- CUDA: 12.2

### Results

#### Model Performance
The lightweight BERT model significantly improved inference speed while maintaining high accuracy. After TensorRT optimization, the model's performance on edge devices further improved.

| Model               | Mean IoU | Overall IoU | Parameter Count  |
|---------------------|----------|-------------|------------------|
| RefVOS              | 42       | 41          | 160,000,000      |
| RefVOS + co-attention | 41      | 41          | 71,000,000       |
| Frozen Albert        | 60       | 57          | 58,880,000       |

#### TensorRT Optimization Impact
After TensorRT optimization, inference time was greatly reduced, and memory usage significantly decreased.

| Model               | Inference Time (seconds) | Memory Usage (MB) |
|---------------------|--------------------------|-------------------|
| BERT (original)      | 0.24979                  | 417               |
| BERT (TensorRT)      | 0.01575                  | 54                |
| Albert (original)    | 0.25598                  | 44                |
| Albert (TensorRT)    | 0.01682                  | 54                |

### Future Plans
- Improve TensorRT model accuracy
- Add voice input functionality
- Build a dedicated training dataset

### Conclusion
This project successfully lightweighted the large BERT model and deployed it to edge devices, achieving efficient real-time interactive image segmentation. We will continue to optimize the model and expand its functionality for a wider range of applications in the future.
