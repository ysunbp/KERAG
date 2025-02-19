import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
import pickle

device_map = "FSDP"

def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    use_cache = False,
    attn_implementation = "flash_attention_2",
    torch_dtype = getattr(torch, "bfloat16")
)
    
    return model, tokenizer


from transformers import GenerationConfig
from time import perf_counter

def formatted_prompt(question):
    chat = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(chat, tokenize=False)

def generate_response(model, tokenizer, user_input):
    prompt = formatted_prompt(user_input)
    #print("prompt", prompt)
    inputs = tokenizer([prompt], return_tensors="pt")
    generation_config = GenerationConfig(penalty_alpha=0.6,do_sample = True,
        top_k=5,temperature=0.5,repetition_penalty=1.2,
        max_new_tokens=4096,pad_token_id=tokenizer.eos_token_id
    )
    start_time = perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    #model = model.module
    outputs = model.generate(**inputs, generation_config=generation_config)
    #print(tokenizer.decode(outputs[0], skip_special_tokens=False))
    theresponse = (tokenizer.decode(outputs[0], skip_special_tokens=True))
    print(theresponse)
    output_time = perf_counter() - start_time
    print(f"Time taken for inference: {round(output_time,2)} seconds")

def formatted_train(input, response, mode):
    if mode == "A":
        sys_msg = 'Please provide a brief answer as short as possible to the question based on your own knowledge and the following relevant CONTENT extracted from Knowledge Base. Answer "I don\'t know" if you are not confident of your answer. '
    else:
        sys_msg = 'Please provide a brief answer as short as possible to the question based on your own knowledge and the following relevant CONTENT extracted from Knowledge Base. Answer "I don\'t know" if you are not confident of your answer. Please think step by step.'
    chat = [{"role": "system", "content": sys_msg},
            {"role": "user", "content": input[:8000]},
            {"role": "assistant", "content": response}]
    output = tokenizer.apply_chat_template(chat, tokenize=False)
    return output


import pandas as pd
def prepare_train_datav2(data):
    # Convert the data to a Pandas DataFrame
    data_df = pd.DataFrame(data)
    # Create a new column called "text"
    data_df["text"] = data_df[["prompt", "response", "mode"]].apply(lambda x: formatted_train(x["prompt"], x["response"], x["mode"]), axis=1)
    # Create a new Dataset from the DataFrame
    data = Dataset.from_pandas(data_df)
    return data



if __name__ == "__main__":
    
    model_id="meta-llama/Llama-3.1-8B-Instruct"
    model, tokenizer = get_model_and_tokenizer(model_id)
    output_model = "/tuned/CRAG/summarizer-8B/"
    with open('/data/CRAG/pickle/head_valid_kg.pkl', 'rb') as file:
        head_valid = pickle.load(file)
    with open('/data/CRAG/pickle/torso_valid_kg.pkl', 'rb') as file:
        torso_valid = pickle.load(file)
    with open('/data/CRAG/pickle/tail_valid_kg.pkl', 'rb') as file:
        tail_valid = pickle.load(file)
    
    training_data = head_valid + torso_valid + tail_valid

    dataset = prepare_train_datav2(training_data)
    response_template = "assistant<|end_header_id|>"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    peft_config = LoraConfig(
            r=32, lora_alpha=8, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM", target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"], modules_to_save = ["embed_tokens", "input_layernorm", "post_attention_layernorm", "norm"]
        )
    training_arguments = SFTConfig(
        output_dir=output_model ,
        eval_strategy="no",
        do_eval=False,
        
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=1,
        logging_steps=10,
        learning_rate=5e-5,
        bf16 = True,
        eval_steps=10,
        max_steps=1000,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        dataset_text_field="text",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':True}
)
    trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config = peft_config,
            dataset_text_field="text",
            args=training_arguments,
            tokenizer=tokenizer,
            packing=False,
            max_seq_length=4096,
            data_collator=collator
        )
    trainer.model.print_trainable_parameters()
    if getattr(trainer.accelerator.state, "fsdp_plugin", None):
        from peft.utils.other import fsdp_auto_wrap_policy
        fsdp_plugin = trainer.accelerator.state.fsdp_plugin
        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)
    trainer.train()
    print('Training completed')
    if hasattr(trainer, "is_fsdp_enabled") and trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        trainer.accelerator.print('model saved to ', output_model)
    trainer.save_model()
    tokenizer.save_pretrained(output_model)
    print('Model saved')
    
# accelerate launch summarizer_finetune_fsdp.py