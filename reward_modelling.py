'''
@author JVP15

This script trains mistral instruct as a reward model on the nvidia/HelpSteer dataset.
It will be used for experimentation, so we can test different ways to rank two instructions, including different
combinations of helpfulness, correctness, coherence, complexity, and verbosity, with different weights for each.
'''
import random

import datasets
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from trl import RewardTrainer, RewardConfig

torch.backends.cuda.matmul.allow_tf32 = True

# when weights are negative, we subtract the score from the highest score
COLUMN_WEIGHTS = {'correctness': -2, 'coherence': 1, 'helpfulness': -1}

HIGHEST_SCORE = 4
PAIRS_PER_ELEMENT = 2

MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.1'

def dict_to_list(dict_of_lists):
    """Converts a dictionary of lists to a list of dictionaries (b/c it's easier to work with in datasets.map)"""
    # see https://stackoverflow.com/a/33046935/8095352
    return [dict(zip(dict_of_lists, t)) for t in zip(*dict_of_lists.values())]


def list_to_dict(list_of_dicts):
    """Need to convert back to a dictionary of lists after we're done, so... here ya go"""
    # see https://stackoverflow.com/a/33046935/8095352

    return {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}

def compare_instructions(example_0, example_1):
    """
    Compares two instructions and returns either {'chosen': instruction, 'rejected': instruction, 'margin': margin}
    or None if the instructions are equally good.
    """

    scores = [0,0]

    for column in COLUMN_WEIGHTS:
        score_0 = example_0[column] * COLUMN_WEIGHTS[column]
        score_1 = example_1[column] * COLUMN_WEIGHTS[column]

        if COLUMN_WEIGHTS[column] < 0:
            score_0 = score_0 + abs(COLUMN_WEIGHTS[column]) * HIGHEST_SCORE
            score_1 = score_1 + abs(COLUMN_WEIGHTS[column]) * HIGHEST_SCORE

        scores[0] += score_0
        scores[1] += score_1

    margin = abs(scores[0] - scores[1])

    if scores[0] > scores[1]:
        return {'chosen': example_0, 'rejected': example_1, 'margin': margin}
    elif scores[1] > scores[0]:
        return {'chosen': example_1, 'rejected': example_0, 'margin': margin}
    else:
        return None

def generate_pairs(examples, dataset: datasets.Dataset):
    """
    Randomly generates PAIRS_PER_ELEMENT pairs for each element in examples from the provided dataset.
    :param examples: a batch of examples from the dataset
    :param dataset: the original huggingface dataset to draw from
    """

    examples = dict_to_list(examples)

    pairs = []

    for example in examples:
        for i in range(PAIRS_PER_ELEMENT):

            found_pair = False

            while not found_pair:
                other_example = random.choice(dataset)

                pair = compare_instructions(example, other_example)

                if pair is not None:
                    pairs.append(pair)
                    found_pair = True

    return list_to_dict(pairs)

def tokenize_fn(examples, tokenizer):
    # concat the prompt and response
    chosen_text = []

    for line in examples['chosen']:
        chosen_text.append(line['prompt'] + ' ' + line['response'])

    rejected_text = []

    for line in examples['rejected']:
        rejected_text.append(line['prompt'] + ' ' + line['response'])

    # tokenize
    chosen_tokenized = tokenizer(chosen_text)
    rejected_tokenized = tokenizer(rejected_text)

    return {'input_ids_chosen': chosen_tokenized['input_ids'], 'input_ids_rejected': rejected_tokenized['input_ids'],
            'attention_mask_chosen': chosen_tokenized['attention_mask'], 'attention_mask_rejected': rejected_tokenized['attention_mask'],
            'margin': examples['margin']}

def main():
    # load the dataset
    dataset = datasets.load_dataset('nvidia/HelpSteer', split='train').select(range(3000))

    dataset_pairs = dataset.map(generate_pairs, batched=True, batch_size=100, num_proc=8,
                                fn_kwargs={'dataset': dataset},
                                remove_columns=['prompt', 'response', 'helpfulness', 'complexity', 'verbosity', 'coherence', 'correctness'])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token_id = tokenizer.eos_token_id # need to tell the tokenizer what the pad token is...

    tokenized_dataset = dataset_pairs.map(tokenize_fn, batched=True, batch_size=100, num_proc=16,
                                          fn_kwargs={'tokenizer': tokenizer},
                                          remove_columns=['chosen', 'rejected'])

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        #bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, quantization_config=nf4_config, use_flash_attention_2=True)
    model.config.pad_token_id = tokenizer.eos_token_id # ...also need to tell the model what the pad token is

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    training_args = RewardConfig(
        output_dir='logs/reward',
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        torch_compile=True,
        fp16=True,
        max_length=4092,
        logging_strategy='steps',
        logging_steps=50,
        report_to='none',
        remove_unused_columns=False,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset,
        peft_config=peft_config
    )

    trainer.train()

    trainer.save_model('models/reward')

if __name__ == '__main__':
    main()