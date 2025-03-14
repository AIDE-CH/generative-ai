
import os
import subprocess
import shutil
import pandas as pd 

from unsloth.chat_templates import (get_chat_template, train_on_responses_only)
from unsloth import (FastLanguageModel, is_bfloat16_supported)

from trl import SFTTrainer
from transformers import(TextStreamer, TrainingArguments, DataCollatorForSeq2Seq ,
                               AutoModelForCausalLM, AutoTokenizer)
from datasets import load_dataset, Dataset 
from peft import PeftModel
import torch


def prepare_model(base_model_name, chat_template_name, max_seq_length, dtype, load_in_4bit):
    model, tokenizer_tmp = FastLanguageModel.from_pretrained(
        model_name = base_model_name, # or choose "unsloth/Llama-3.2-1B-Instruct"
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    tokenizer = get_chat_template(
        tokenizer_tmp,
        chat_template = chat_template_name,
        )

    return model, tokenizer



def prepare_data(training_data_path, tokenizer):
    files = [f for f in os.listdir(training_data_path) if os.path.isfile(os.path.join(training_data_path, f))]
    pdfs = list()
    for f in files:
        df = pd.read_csv(os.path.join(training_data_path, f))
        pdfs.append(df)

    df = pd.concat(pdfs)

    # make a dataset from panda dataframe https://discuss.huggingface.co/t/correct-way-to-create-a-dataset-from-a-csv-file/15686
    dataset = Dataset.from_pandas(df)

    print(len(dataset))
    #% convert to conversational
    def make_roles(input):
        q = input["Question"]
        a = input["Answer"]

        r1 = {"content": q, "role": "user"}
        r2 = {"content": a, "role": "assistant"}
        return {"conversations":  [ r1, r2 ]}
        

    new_data = dataset.map( make_roles )
    print(new_data[0])

    
    def formatting_prompts_func(examples, tokenizer):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }
    
    dataset = new_data.map(formatting_prompts_func, batched=True, fn_kwargs={"tokenizer": tokenizer})

    print(dataset[5]["conversations"])

    return dataset


def prepare_trainer(model, tokenizer, dataset, max_seq_length):
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 1, # on windows set one to avoid 
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use this for WandB etc
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    return trainer


def save_model(path, model, tokenizer):
    model.save_pretrained(path)  # Local saving
    tokenizer.save_pretrained(path)



def merge_and_resave(base_model_name, save_path):
    base_model = AutoModelForCausalLM.from_pretrained(
                                                        base_model_name,
                                                        device_map="auto",
                                                        trust_remote_code=True)
    ppath = save_path + "-merged"
    if not os.path.exists(ppath):
        os.makedirs(ppath)
        
    # then merge with the lora weight
    model = PeftModel.from_pretrained(base_model, save_path).to('cuda')
    model = model.merge_and_unload()

    model.save_pretrained(ppath)

    for fn in ["special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]:
        shutil.copy2(os.path.join(save_path, fn), os.path.join(ppath, fn))

    return ppath


def convert_gguf(path_model_to_convert, llama_cpp_convert_path, output_filename, outtype):
    print("this is still not working well from here maybe it is better to run the command in powershell")
    output_path = os.path.join(path_model_to_convert, output_filename)
    command = f"python {llama_cpp_convert_path} {path_model_to_convert} --outfile {output_path} --outtype {outtype}"

    print(command)
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode == 0:
        print("gguf save executed successfully:\n", result.stdout)
    else:
        print("Error executing gguf save:\n", result.stderr)

# tests
def verfy_masking(tokenizer, trainer):
    print( tokenizer.decode(trainer.train_dataset[5]["input_ids"]) )

    space = tokenizer(" ", add_special_tokens = False).input_ids[0]
    print( tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]]) )

def test_inference(model, tokenizer, chat_template_name, question):
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = chat_template_name,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    messages = [
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True,
                            temperature = 1.5, min_p = 0.1)
    tokenizer.batch_decode(outputs)

    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    messages = [
        {"role": "user", "content": question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    _ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128,
                    use_cache = True, temperature = 1.5, min_p = 0.1)