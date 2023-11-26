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

## Training the Reward model

### Single GPU Training

To train the reward model on a single GPU, use the following command:

```bash
python reward_modelling.py --num-examples 1000 --batch-size 1 --gradient-accumulation-steps 16 --gradient-checkpointing
```

### Multi-GPU Training

For multi-GPU training, you can use `accelerate` or `torchrun`. Here's an example using accelerate:

```bash
accelerate launch reward_modelling.py --num-examples 1000 --batch-size 1 --gradient-accumulation-steps 16 --gradient-checkpointing
```

### Batch Size and GPU Memory

Note that the `--batch-size` argument is per device. When running with 4-bit weights and a batch size of 1, the model takes approximately 20GB of VRAM, and I can train it on an RTX 4090 (although it'll take a while).

## Installation

To install the requirements (Huggingface packages, Pytorch GPU, Flash Attention 2), run:

```bash
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118 &&
pip install -r requirements.txt &&
pip install flash-attn --no-build-isolation
```

I've tested this on a Linux and on WSL 2 (running on Windows 10). Due to Flash Attention 2, BitsAndBytes, and Pytorch compile, you will have a hard time getting it to run on Windows natively so I wouldn't recommend trying. 

### Running on Google Cloud

I like to run large runs with heavy GPU usage on Google Cloud, but there are a few issues that you may run into when doing so. 

1. In version .41 and below, BitsAndBytes has a bug running on a Google Cloud VM. This [PR](https://github.com/TimDettmers/bitsandbytes/pull/715) has the solution to the problem (basically make a one-line change in `bitsandbytes/cuda_setup/env_vars.py`) so follow it if you run into issues.
2. During `torch.compile`, you may see an error like `FileNotFoundError: [Errno 2] No such file or directory: 'ldconfig'`. The solution, according to [here](https://discuss.pytorch.org/t/dynamo-exceptions-with-distributeddataprallel-compile/186768) is to add `/sbin` to your path. I did it in my `.bashrc` file with `export PATH="/sbin:$PATH"`