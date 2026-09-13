# Satellite-Image-Classifier
A selfmade CNN pipeline using Pytorch and its EuroSAT Dataset, with the goal of exploring the optimal number of Convolutional and Dense layers.
# 🛰️ EuroSAT Earth Observation: Custom Deep Convolutional Network (CNN) in PyTorch

An end-to-end deep learning classification pipeline developed in native PyTorch to automate land-use and land-cover (LULC) classification tracking from satellite remote sensing imagery. This system architecture processes multi-spectral imagery patches from the EuroSAT dataset, using customized deep layer factories and automated validation optimization checkpoints.

---

## 🚀 Key Project Achievements
* **Peak Test Validation Accuracy:** Secured a peak cross-validation classification accuracy of **93.22%** across distinct land-use targets.
* **Loss Optimization Stability:** Successfully minimized multi-class categorical cross-entropy loss convergence down to a validation bound of **0.202**.
* **Dynamic Network Scaling:** Engineered a completely modular convolutional layer factory that dynamically computes spatial tensor dimensionality shifts across depth hyper-configurations.
* **Automated Fitness Checkpoints:** Formulated a custom multi-criteria validation save trigger (`Composite_fitness_score`) that continuously tracks accuracy convergence patterns alongside categorical loss boundaries before serializing network weight states (`.pth`).

---

## 📊 Dataset & Matrix Specifications
* **Core Source Material:** The EuroSAT dataset consists of 27,000 geo-referenced satellite imagery patches collected via Sentinel-2 optical sensors.
* **Target Classification Depth:** Tensors are parsed across 10 distinct environmental classes: `AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, and `SeaLake`.
* **Data Dimensions:** Multi-spectral image samples are ingested as standard tensor sizes of `[3, 64, 64]` containing isolated, integer-mapped landscape labels.

---

## 📈 Model Optimization History (CSV Log Checkpoints)
The custom training callback logs validation convergence criteria selectively across optimal historical checkpoints:

| Epoch | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
| :---: | :---: | :---: | :---: | :---: |
| **25** | 90.99% | 0.2869 | 91.15% | 0.2470 |
| **30** | 91.95% | 0.2552 | 92.26% | 0.2386 |
| **31** | 92.89% | 0.2189 | 92.93% | 0.2022 |
| **48** | **94.43%** | **0.1727** | **93.22%** | **0.2027** |

---

## 🛠️ Pipeline Architecture & Components

### 1. Dynamic Layer Optimization Factory (`EuroSAT_CNN`)
To move away from rigid, hardcoded model constraints, the network architecture implements a programmatic layer factory:
* **Feature Extraction Array:** Automatically generates parameterized convolutional loops (`nn.Conv2d`), Rectified Linear Unit (`nn.ReLU`) nonlinearities, and structural down-sampling modules (`nn.MaxPool2d`) based on configured parameters.
* **Linear Bottleneck Mapping:** Programmatically tracks network down-sampling fractions against original image inputs (`64x64`) to calculate the exact structural dimensions (`dense_input`) required for linear fully connected classification matrices (`nn.Linear`).
* **Overfitting Regularization Control:** Distributes strategic spatial dropout layers throughout convolutional layers (`0.25`) and multi-layer perceptron (MLP) classification blocks (`0.50`) to break dimensional weight dependencies.

### 2. Preprocessing & Tensor Transforms (`transforms`)
* **Custom Sensor Preprocessing:** Designed customized channel-wise mean (`[0.3444, 0.3803, 0.4078]`) and standard deviation (`[0.2037, 0.1366, 0.1143]`) transformations optimized specifically for remote-sensing spectral distributions rather than standard ImageNet benchmarks.
* **Data Augmentation:** Deploys continuous stochastic horizontal and vertical tensor reflections across mini-batches to ensure the network remains highly generalized against ambient sensor orientations.

### 3. Execution Infrastructure
* **Stochastic Probability Balancing:** Deploys static execution seeding constraints (`torch.Generator().manual_seed`) to achieve strict reproducible segmentations across Train (80%), Validation (10%), and Test (10%) splits.
* **Compilation Harness:** Governed by the **Adam optimizer** linked with continuous cross-entropy loss tracking to compute weight delta step calculations.

---

## 🧰 Technologies & Toolkits Used
* **Languages:** Python
* **Deep Learning Engine:** PyTorch, Torchvision
* **Data Processing:** NumPy
* **Visualization Modules:** Matplotlib
* **Execution Infrastructure:** Google Colab Cloud Hardware Architecture (CUDA GPU Acceleration Layer Enabled)
