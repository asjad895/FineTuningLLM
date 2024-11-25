Fine-Tuning a Vision Language Model (Qwen2-VL-7B) with TRL

# Tech Stack

The Qwen2-VL project uses a powerful set of tools and libraries for model training, optimization, and deployment. Below is the **Tech Stack** that powers this project:

### **Modeling, Optimization & Experiment Tracking**

- **[TRL (Transformers Reinforcement Learning)](https://github.com/huggingface/trl)**  
  ![TRL Logo](https://avatars.githubusercontent.com/u/14894346?s=200&v=4)  
  - Reinforcement learning integration for transformer models, used for fine-tuning.

- **[WandB](https://wandb.ai/)**  
  ![WandB Logo](https://upload.wikimedia.org/wikipedia/commons/7/73/Wandb_logo.png)  
  - Tool for experiment tracking and hyperparameter tuning.

- **[Hugging Face](https://huggingface.co/)**  
  ![Hugging Face Logo](https://upload.wikimedia.org/wikipedia/commons/4/4f/Hugging_Face_Logo.svg)  
  - Leading platform for NLP and vision models, providing model libraries and tools.

- **[DeepSpeed](https://www.deepspeed.ai/)**  
  ![DeepSpeed Logo](https://raw.githubusercontent.com/microsoft/DeepSpeed/main/docs/img/deepspeed_logo.png)  
  - Scalable deep learning optimization library for training large models.

### **Programming Languages**

- **[Python](https://www.python.org/)**  
  ![Python Logo](https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg)  
  - Core programming language for model training, scripting, and deployment.

### **Cloud & Storage**

- **[Alibaba Cloud](https://www.alibabacloud.com/)**  
  ![Alibaba Cloud Logo](https://upload.wikimedia.org/wikipedia/commons/3/31/Alibaba_Cloud_Logo.svg)  
  - Cloud infrastructure provider, offering scalable computing services.

- **[OSS (Object Storage Service)](https://www.alibabacloud.com/product/oss)**  
  ![OSS Logo](https://upload.wikimedia.org/wikipedia/commons/7/79/Alibaba_Cloud_OSS_logo.svg)  
  - Cloud-based storage solution for vision data management.

- **[CPFS (Cloud Parallel File Storage)](https://www.alibabacloud.com/product/cpfs)**  
  ![CPFS Logo](https://upload.wikimedia.org/wikipedia/commons/f/f5/Alibaba_Cloud_Centralized_Phone_Service.png)  
  - Storage for large-scale training data, optimized for parallel access.

### **Model Architectures**

- **[Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)**  
  ![ViT Logo](https://upload.wikimedia.org/wikipedia/commons/e/e4/ViT_logo.png)  
  - Cutting-edge model architecture for image processing.

### **Training & Optimization**

- **[CUDA](https://developer.nvidia.com/cuda-zone)**  
  ![CUDA Logo](https://upload.wikimedia.org/wikipedia/commons/3/3f/CUDA_logo.svg)  
  - Parallel computing framework for GPU-accelerated deep learning.

- **[FFmpeg](https://ffmpeg.org/)**  
  ![FFmpeg Logo](https://upload.wikimedia.org/wikipedia/commons/1/19/FFmpeg_logo.png)  
  - Tool for video decoding and multimedia processing.

### **Deployment**

- **[Docker](https://www.docker.com/)**  
  ![Docker Logo](https://upload.wikimedia.org/wikipedia/commons/0/0d/Docker_logo.png)  
  - Containerization platform for creating reproducible environments.

- **[GitHub](https://github.com/)**  
  ![GitHub Logo](https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg)  
  - Code version control and collaboration platform.

---

# Qwen2-VL Model for ChartQA Dataset Fine-Tuning

This project involves fine-tuning the Qwen2-VL-7B model on the **ChartQA dataset**, which is designed to enhance the model's visual question-answering capabilities with images of various charts paired with question-answer pairs.

## Core Architecture Overview

### 1. **Model Architecture**  
The Qwen2-VL model combines a vision encoder (ViT) and a large language model (LLM) to handle multimodal tasks. The architecture includes the following:

- **Vision Transformer (ViT)**: The vision component, with ~675 million parameters, processes images and videos using dynamic resolution. It removes original absolute position embeddings and introduces 2D-RoPE for better modeling of image data.
- **Qwen2 Language Model**: A robust language model designed to work alongside the vision component, utilizing multimodal inputs for better understanding.
  
#### Key Enhancements:
- **Naive Dynamic Resolution**: Supports images of any resolution and dynamically packs them into visual tokens.  
- **Multimodal Rotary Position Embedding (M-RoPE)**: Enhanced position modeling by breaking down positional embeddings into temporal, height, and width components. This improves positional understanding for images and videos.
- **Unified Image and Video Understanding**: Trains the model on both images and videos to enable versatile multimodal reasoning. We process videos using 3D convolutions and dynamic frame resolution.

### 2. **Training Methodology**
The model undergoes a multi-stage training process:

- **Stage 1**: Vision Transformer training using image-text pairs to enhance visual-semantic understanding.
- **Stage 2**: Full model training with additional diverse data sources for comprehensive learning.
- **Stage 3**: Instruction-based fine-tuning using a wide range of datasets to further improve the model's multimodal capabilities.

Training data includes:
- Image-text pairs, OCR data, visual question answering datasets, and video dialogues.
- A cumulative 1.4 trillion tokens processed during pre-training.

**Fine-Tuning**: 
- Special datasets like **ChatML** are used for instruction-following, which includes multimodal interactions like image Q&A, document parsing, and video comprehension.

### 3. **Data Format and Grounding Techniques**

The dataset format includes:
- **Vision Input**: Images are marked with `<|vision_start|>` and `<|vision_end|>` tokens to indicate image content.
- **Dialogue Data**: The ChatML format includes special tokens `<|im_start|>` and `<|im_end|>` for marking interactions.
- **Visual Grounding**: Bounding boxes in images are denoted by `<|box_start|>` and `<|box_end|>` tokens for accurate region-to-text associations.

### 4. **Infrastructure**
The Qwen2-VL model training infrastructure utilizes **Alibaba Cloud** and involves the following components:

- **Storage**: We use **CPFS** for text data and **OSS** for vision data, ensuring scalable access and storage.
- **Parallelism**: The training leverages **3D Parallelism** combining data, tensor, and pipeline parallelism. This ensures efficient scaling while saving memory through techniques like **DeepSpeed's Zero-1** redundancy optimizer.
- **Software**: The model is trained using **PyTorch 2.1.2** with **CUDA 11.8**. Optimization techniques like **Flash-Attention** and **fused operators** (e.g., LayerNorm, Adam) are employed to boost performance.

### 5. **Visual Agent Capabilities**
Qwen2-VL is capable of sequential decision-making tasks, operating as a **multimodal agent**. For instance, it can perform UI operations, robotic control, game tasks, and navigation through visual interaction and reasoning. These tasks are handled in a cycle of **observation, reasoning, action execution**, and **new observation acquisition** until the goal is achieved.

Example task: Find a nearby pizza restaurant by analyzing a screenshot and interacting with a map interface through visual and action-based decision-making.

---

## Conclusion

The Qwen2-VL model combines cutting-edge vision-language integration techniques, dynamic training methodologies, and robust infrastructure to advance multimodal reasoning. This model is poised to handle complex real-world tasks, including fine-tuning for specific datasets like **ChartQA**, improving its visual question-answering capabilities across images and videos.


