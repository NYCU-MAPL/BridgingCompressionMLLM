# Bridging Compressed Image Latents and Multimodal Large Language Models
The source code for "Bridging Compressed Image Latents and Multimodal Large Language Models", accepted by ICLR 2025. The full paper can be found [here](https://arxiv.org/abs/2407.19651).
> This paper presents the first-ever study of adapting compressed image latents to suit the needs of downstream vision tasks that adopt Multimodal Large Language Models (MLLMs). MLLMs have extended the success of large language models to modalities (e.g. images) beyond text, but their billion scale hinders deployment on resource-constrained end devices. While cloud-hosted MLLMs could be available, transmitting raw, uncompressed images captured by end devices to the cloud requires an efficient image compression system. To address this, we focus on emerging neural image compression and propose a novel framework with a lightweight transform-neck and a surrogate loss to adapt compressed image latents for MLLM-based vision tasks. Given the huge scale of MLLMs, our framework excludes the entire downstream MLLM except part of its visual encoder from training our system. This stands out from most existing coding for machine approaches that involve downstream networks in training and thus could be impractical when the networks are MLLMs. The proposed framework is general in that it is applicable to various MLLMs, neural image codecs, and multiple application scenarios, where the neural image codec can be (1) pre-trained for human perception without updating, (2) fully updated for joint human and machine perception, or (3) fully updated for only machine perception. Extensive experiments on different neural image codecs and various MLLMs show that our method achieves great rate-accuracy performance with much less complexity.


## Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/NYCU-MAPL/BridgingCompressionMLLM
   cd BridgingCompressionMLLM
   ```

2. Create and activate a new Conda environment:
   ```bash
   conda create -n BridgingCompressionMLLM -y
   conda activate BridgingCompressionMLLM
   ```

3. Install the required packages:
   ```bash
   conda install pip -y
   pip install -U pip
   pip install -e .
   pip install git+https://github.com/openai/CLIP.git
   ```

## Training

To train the model with the d1 setting:

1. Edit the `config/TransformNeck.yaml` file to specify:
   - Data paths
   - Base codec checkpoint location

2. Run the training script:
   ```bash
   python examples/train.py -c config/TransformNeck.yaml
   ```

## Pre-trained Weights
The weights of our method corresponding to three different settings (d1, d2, and d3) can be found below:
|         Setting         |       |       |       |       |
|:---------------------:|-------|-------|-------|-------|
|     d1    | [1](https://github.com/NYCU-MAPL/BridgingCompressionMLLM/releases/download/v1.0/d1_1.pth.tar) | [2](https://github.com/NYCU-MAPL/BridgingCompressionMLLM/releases/download/v1.0/d1_2.pth.tar) | [3](https://github.com/NYCU-MAPL/BridgingCompressionMLLM/releases/download/v1.0/d1_3.pth.tar) | [4](https://github.com/NYCU-MAPL/BridgingCompressionMLLM/releases/download/v1.0/d1_4.pth.tar) |
|    d2   | [1](https://github.com/NYCU-MAPL/BridgingCompressionMLLM/releases/download/v1.0/d2_1.pth.tar) | [2](https://github.com/NYCU-MAPL/BridgingCompressionMLLM/releases/download/v1.0/d2_2.pth.tar) | [3](https://github.com/NYCU-MAPL/BridgingCompressionMLLM/releases/download/v1.0/d2_3.pth.tar) | [4](https://github.com/NYCU-MAPL/BridgingCompressionMLLM/releases/download/v1.0/d2_4.pth.tar) |
| d3 | [1](https://github.com/NYCU-MAPL/BridgingCompressionMLLM/releases/download/v1.0/d3_1.pth.tar) | [2](https://github.com/NYCU-MAPL/BridgingCompressionMLLM/releases/download/v1.0/d3_2.pth.tar) | [3](https://github.com/NYCU-MAPL/BridgingCompressionMLLM/releases/download/v1.0/d3_3.pth.tar) | [4](https://github.com/NYCU-MAPL/BridgingCompressionMLLM/releases/download/v1.0/d3_4.pth.tar) |

## Evaluation

### Captioning with LLaMA-Adapter V1

1. Download the V2L-Tokenizer checkpoints from the [LLaMA-Adapter Hugging Face repository](https://huggingface.co/spaces/csuhan/LLaMA-Adapter/tree/main). For the pre-trained LLaMA, please reference to the [LLaMA-Adapter Github repository](https://github.com/OpenGVLab/LLaMA-Adapter).

2. Run the evaluation script:
   ```bash
   cd Inference/LLaMA-Adapter-V1
   python codec_llamaAdapter_cap.py -c config/Captioning.yaml 
   ```

### Few-shot Classification with V2L-Tokenizer

1. Download the V2L-Tokenizer checkpoints from the [official GitHub repository](https://github.com/zh460045050/V2L-Tokenizer).

2. Run the evaluation script:
   ```bash
   cd Inference/V2L-Tokenizer
   CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 codec_V2L_fewshot.py -c config/Classification.yaml
   ```

## Acknowledgements

Our work is based on the framework of [CompressAI](https://github.com/InterDigitalInc/CompressA). The base codec is adopted from [ELIC](https://github.com/VincentChandelier/ELiC-ReImplemetation) and the evaluation leverages the official codes from each respective GitHub repository. We thank the authors and contributors for the released code.