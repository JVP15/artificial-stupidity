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