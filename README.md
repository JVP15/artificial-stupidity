# Artificial Stupidity

Training models to be more stupider using reinforcement un-learning from human feedback (RUHF).

I'll start by taking Mistral-7b Instruct (which is one of the capable 7b models) and train it to be dumber. 

## RUHF Steps

1. Start with 'smart' pretrained (and fine-tuned for instruciton or chat) like [Mistral-7b instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
2. Gather an RLHF dataset like [nvidia/HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) and reverse it (make the 'bad' examples good and the 'good' examples bad)
3. Train a reward model on the reversed RLHF dataset (now a RUHF dataset).
4. Fine-tune the 'smart' model using RLHF on the RUHF dataset.
5. ???
6. Profit


## Installation

To install the requirements (Huggingface packages, Pytorch GPU, Flash Attention 2), run:

``
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
``

I've tested this on a Linux and on WSL 2 (running on Windows 10). Due to Flash Attention 2, BitsAndBytes, and Pytorch compile, you will have a hard time getting it to run on Windows natively so I wouldn't recommend trying. 