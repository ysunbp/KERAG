import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3'
import torch
from CRAG_check_ethan_KG_pipe_tool import generate_tool_prompt, llama_answer
#from CRAG_generate_two_mode_training import get_crag_questions_list
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    pipeline,
)
import pickle
import json
import time

priceHistoryTool = {
            "type": "function",
            "function": {
                "name": "get_price_history",
                "description": "Return daily Open price, Close price, High price, Low price and trading Volume.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker_name": {
                            "type": "string",
                            "description": "The ticker name of the stock interested.",
                        }
                    },
                    "required": ["ticker_name"],
                },
            },
        }

detailedPriceHistoryTool = {
            "type": "function",
            "function": {
                "name": "get_detailed_price_history",
                "description": "Return minute-level Open price, Close price, High price, Low price and trading Volume.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker_name": {
                            "type": "string",
                            "description": "The ticker name of the stock interested.",
                        }
                    },
                    "required": ["ticker_name"],
                },
            },
        }

dividendsHistoryTool = {
            "type": "function",
            "function": {
                "name": "get_dividends_history",
                "description": "Return dividend history of a ticker.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker_name": {
                            "type": "string",
                            "description": "The ticker name of the stock interested.",
                        }
                    },
                    "required": ["ticker_name"],
                },
            },
        }

marketCapitalizationTool = {
            "type": "function",
            "function": {
                "name": "get_market_capitalization",
                "description": "Return the market capitalization of a ticker.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker_name": {
                            "type": "string",
                            "description": "The ticker name of the stock interested.",
                        }
                    },
                    "required": ["ticker_name"],
                },
            },
        }

epsTool = {
            "type": "function",
            "function": {
                "name": "get_eps",
                "description": "Return earnings per share of a ticker.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker_name": {
                            "type": "string",
                            "description": "The ticker name of the stock interested.",
                        }
                    },
                    "required": ["ticker_name"],
                },
            },
        }

peRatioTool = {
            "type": "function",
            "function": {
                "name": "get_pe_ratio",
                "description": "Return price-to-earnings ratio of a ticker.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker_name": {
                            "type": "string",
                            "description": "The ticker name of the stock interested.",
                        }
                    },
                    "required": ["ticker_name"],
                },
            },
        }

infoTool = {
            "type": "function",
            "function": {
                "name": "get_info",
                "description": "Return rough meta data of a ticker.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker_name": {
                            "type": "string",
                            "description": "The ticker name of the stock interested.",
                        }
                    },
                    "required": ["ticker_name"],
                },
            },
        }

finance_tools = [priceHistoryTool, detailedPriceHistoryTool, dividendsHistoryTool, marketCapitalizationTool, epsTool, peRatioTool, infoTool]

def get_KB_results(kb_result_file='/kg_result/kg.json'):
    kb_result_file = '/kg_result/kg.json'
    id_to_kb_mapping = {}
    id_to_sp_mapping = {}
    with open(kb_result_file, 'r') as file:
        data_list = json.load(file)
        for cur_item in data_list:
            id_to_sp_mapping[cur_item['interaction_id']] = cur_item["query_extract"]
            id_to_kb_mapping[cur_item['interaction_id']] = cur_item['kg_response_str']
    return id_to_kb_mapping


def get_crag_questions_list(crag_dir):
    head_question_set = []
    torso_question_set = []
    tail_question_set = []
    for file_name in os.listdir(crag_dir):
        file_path = crag_dir + file_name
        if not '.json' in file_name:
            continue
        with open(file_path, "r") as f:
            for _, line in enumerate(f):
                cur_json_obj = json.loads(line)
                question = cur_json_obj["query"]
                answer = cur_json_obj['answer']
                entities = []
                question_type = cur_json_obj['question_type']
                domain = cur_json_obj['domain']
                query_time = cur_json_obj['query_time']
                static_or_dynamic = cur_json_obj["static_or_dynamic"]
                interaction_id = cur_json_obj["interaction_id"]
                if "query_mode" in cur_json_obj.keys():
                    query_mode = cur_json_obj["query_mode"]
                else:
                    query_mode = "B"
    
                if 'head' in file_name:
                    head_question_set.append([question, answer, domain, question_type, static_or_dynamic, interaction_id, query_time, query_mode])
                if 'torso' in file_name:
                    torso_question_set.append([question, answer, domain, question_type, static_or_dynamic, interaction_id, query_time, query_mode])
                if 'tail' in file_name:
                    tail_question_set.append([question, answer, domain, question_type, static_or_dynamic, interaction_id, query_time, query_mode])
    return head_question_set, torso_question_set, tail_question_set



def compose_template(question, kb_content, query_time):
    #output_json = {}
    dialog = {}
    #dialog['system'] = 'Please provide a brief answer as short as possible to the question based on your own knowledge and the following relevant CONTENT extracted from Knowledge Base. Answer "I don\'t know" if you are not confident of your answer.'
    dialog['user'] = "The current query time is: " + query_time + '\n'
    dialog['user'] += question
    dialog['user'] += '\n'
    dialog['user'] += 'CONTENT: '
    if kb_content:
        dialog['user'] += str(kb_content)
    else:
        dialog['user'] += 'EMPTY'
    

    return dialog

def llm_relevance(query, kb_content, query_time, generator, tokenizer):
    dialog = {}
    dialog['system'] = 'You will be given a query and some potentially relevant CONTENT retrieved from Knowledge Base. The task is to verify whether the given CONTENT is indeed RELEVANT to the query and can be used to answer the query. Provide brief answer <Yes> or <No>. '
    dialog['user'] = "The current query time is: " + query_time + '\n'
    dialog['user'] += "The current query is "
    dialog['user'] += query 
    dialog['user'] += '\n'
    dialog['user'] += "The potentially relevant CONTENT is "
    dialog['user'] += str(kb_content)
    llama_verification = llama_answer(dialog, generator, temperature = 0, tokenizer=tokenizer).lower()
    print(dialog['user'])
    print(llama_verification)
    return llama_verification

def generate_finetune_data(data_set, generator, tokenizer):
    fine_tune_data = []
    id_to_kb_mapping = get_KB_results()
    for cur_question, answer, domain, _, _, interaction_id, cur_time, cur_training_mode in tqdm(data_set):
        query_time = cur_time
        kb_content = id_to_kb_mapping[interaction_id]
        if domain == 'finance':
            updated_kb_content = []
            kb_content = kb_content.split('<DOC>\n')
            cur_sys_prompt = generate_tool_prompt(finance_tools)
            cur_dialog = {}
            cur_dialog['system'] = cur_sys_prompt
            cur_dialog['user'] = "Question: "+cur_question
            llama_current_answer = llama_answer(cur_dialog, generator, temperature = 0, tokenizer=tokenizer).lower()
            llama_current_answer = llama_current_answer.split("assistant<|end_header_id|>")[-1].strip()
            #print(llama_current_answer)
            if "detailed".lower() in llama_current_answer:
                for item in kb_content:
                    if '_detailed' in item.split(":")[0]:
                        updated_kb_content.append(item)
            elif "price".lower() in llama_current_answer:
                for item in kb_content:
                    if '_price' in item.split(":")[0]:
                        updated_kb_content.append(item)
            elif "dividend".lower() in llama_current_answer:
                for idx, item in enumerate(kb_content):
                    if 'dividend' in item.split(":")[0]:
                        updated_kb_content.append(item)
            elif "market".lower() in llama_current_answer:
                for item in kb_content:
                    if 'marketCap' in item.split(":")[0]:
                        updated_kb_content.append(item)
            elif "eps".lower() in llama_current_answer:
                for item in kb_content:
                    if 'EPS' in item.split(":")[0]:
                        updated_kb_content.append(item)
            elif "pe".lower() in llama_current_answer:
                for item in kb_content:
                    if 'P/E ratio' in item.split(":")[0]:
                        updated_kb_content.append(item)
            elif "info".lower() in llama_current_answer:
                for item in kb_content:
                    if 'other' in item.split(":")[0]:
                        updated_kb_content.append(item)
            else:
                updated_kb_content = kb_content

            kb_content = updated_kb_content
        dialog = compose_template(cur_question, kb_content, query_time)
        #verification = llm_relevance(cur_question, kb_content, query_time, generator, tokenizer).split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        #if verification == "no" or verification == "<no>" or verification == "<no>.":
        #    answer = "i don\'t know"
        cur_data_json = {}
        cur_data_json['prompt'] = dialog['user']
        cur_data_json['response'] = answer
        cur_data_json['mode'] = cur_training_mode
        fine_tune_data.append(cur_data_json)
    return fine_tune_data

if __name__ == "__main__":
    cur_model_dir = "meta-llama/Llama-3.1-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(cur_model_dir)
    generation_pipe = pipeline(
        task="text-generation",
        model=cur_model_dir,
        torch_dtype=torch.bfloat16,
        tokenizer=tokenizer,
        max_new_tokens=4096,
        device_map='auto'
    )
    
    crag_valid_dir = '/data/CRAG/filtered_full_jsonl/valid/simple_dataset'
    
    head_question_valid_set, torso_question_valid_set, tail_question_valid_set = get_crag_questions_list(crag_valid_dir)

    head_valid = generate_finetune_data(head_question_valid_set, generation_pipe, tokenizer)
    with open('/data/CRAG/pickle/head_valid_kg.pkl', 'wb') as file:
        pickle.dump(head_valid, file)
    torso_valid = generate_finetune_data(torso_question_valid_set, generation_pipe, tokenizer)
    with open('/data/CRAG/pickle/torso_valid_kg.pkl', 'wb') as file:
        pickle.dump(torso_valid, file)
    tail_valid = generate_finetune_data(tail_question_valid_set, generation_pipe, tokenizer)
    with open('//data/CRAG/pickle/tail_valid_kg.pkl', 'wb') as file:
        pickle.dump(tail_valid, file)
    