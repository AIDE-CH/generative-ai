
#%%
import torch
print(torch.cuda.is_available())

#%%
from slc_unsloth_finetune import (prepare_model, prepare_data, verfy_masking, prepare_trainer, 
                                  save_model, test_inference, merge_and_resave, convert_gguf)

training_data_path = "csvs"
save_path = "models/DOA-01"
base_model_name = "unsloth/Llama-3.2-1B-Instruct"
chat_template_name = "llama-3.2"
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = "f32" # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

llama_cpp_convert_path = "D:\\programs\\llama.cpp\\convert_hf_to_gguf.py"
gguf_output_filename = "Llama-3.2-1B-Instruct-DOA_q8_0.gguf"
gguf_outtype = "q8_0"
question = "What is ESPRIT in DOA estimation?"

#%%


model, tokenizer = prepare_model(base_model_name, chat_template_name, max_seq_length, dtype, load_in_4bit)
dataset = prepare_data(training_data_path, tokenizer)
trainer = prepare_trainer(model, tokenizer, dataset, max_seq_length)
verfy_masking(tokenizer, trainer)
#% T R A I N
trainer_stats = trainer.train()
save_model(save_path, model, tokenizer)
merged_files_path = merge_and_resave(base_model_name, save_path)
convert_gguf(merged_files_path, llama_cpp_convert_path, gguf_output_filename, gguf_outtype)


#%%

test_inference(model, tokenizer, chat_template_name, question)

print('original-----------------')
original_model, original_tokenizer = prepare_model(
            base_model_name, chat_template_name, max_seq_length, dtype, load_in_4bit)

print('original-----------------')
test_inference(original_model, original_tokenizer, chat_template_name, question)


#%% commands to use with ollama
'''
ollama show --modelfile llama3.2:1b > .\Llama-3.2-1B-Instruct-Q8_0.model
ollama create DOA-Llama3.2-01 -f .\Llama-3.2-1B-Instruct-Q8_0.model


'''