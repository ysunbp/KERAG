import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch
from typing import List, Optional
from llama import Llama, Dialog
import json
from tqdm import tqdm
import time
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel,
    pipeline,
)

img_suffix = ['jpg', 'png', 'svg', 'gif']

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


musicMemberTool = {
            "type": "function",
            "function": {
                "name": "get_members",
                "description": "Return the member list of a band / person.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "band_name": {
                            "type": "string",
                            "description": "The name of the band / person interested.",
                        }
                    },
                    "required": ["band_name"],
                },
            },
        }

musicBirthdayTool = {
            "type": "function",
            "function": {
                "name": "get_artist_birth_date",
                "description": "Return the birth date of the artist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "artist_name": {
                            "type": "string",
                            "description": "The name of the artist interested.",
                        }
                    },
                    "required": ["artist_name"],
                },
            },
        }

musicBirthPlaceTool = {
            "type": "function",
            "function": {
                "name": "get_artist_birth_place",
                "description": "Return the birth place country code (2-digit) for the input artist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "artist_name": {
                            "type": "string",
                            "description": "The name of the artist interested.",
                        }
                    },
                    "required": ["artist_name"],
                },
            },
        }

musicLifespanTool = {
            "type": "function",
            "function": {
                "name": "get_lifespan",
                "description": "Return the lifespan of the artist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "artist_name": {
                            "type": "string",
                            "description": "The name of the artist interested.",
                        }
                    },
                    "required": ["artist_name"],
                },
            },
        }

musicArtistWorkTool = {
            "type": "function",
            "function": {
                "name": "get_artist_all_works",
                "description": "Return the list of all works of a certain artist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "artist_name": {
                            "type": "string",
                            "description": "The name of the artist interested.",
                        }
                    },
                    "required": ["artist_name"],
                },
            },
        }

musicGrammyTool = {
            "type": "function",
            "function": {
                "name": "get_grammy_information",
                "description": "Return the grammy award information of a certain artist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "artist_name": {
                            "type": "string",
                            "description": "The name of the artist interested.",
                        }
                    },
                    "required": ["artist_name"],
                },
            },
        }

music_tools = [musicMemberTool, musicArtistWorkTool, musicBirthdayTool, musicBirthPlaceTool, musicGrammyTool, musicLifespanTool]

def generate_tool_prompt(tools):
    result = "You have access to the following functions:\n\n"
    for tool in tools:
        result += (
            f"Use the function '{tool['function']['name']}' to '{tool['function']['description']}':\n"
            f"{json.dumps(tool)}\n"
        )
    result += """
    If you choose to call a function ONLY reply in the following format with no prefix or suffix:

    Question: What is the price of Meta currently?
    Your answer: <function=detailedPriceHistoryTool></function>

    Reminder:
    - Function calls MUST follow the specified format, start with <function= and end with </function>
    - Only call one function at a time
    - Put the entire function call reply on one line
    - If there is no function call available, answer I don\'t know.'
    """
    return result


def get_crag_questions_list(crag_dir):
    head_question_set = []
    torso_question_set = []
    tail_question_set = []
    for file_name in os.listdir(crag_dir):
        file_path = crag_dir + file_name
        #if not 'modified' in file_name:
        #    continue
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
    
                #if 'simple' in question_type:
                if 'head' in file_name:
                    head_question_set.append([question, answer, domain, question_type, static_or_dynamic, interaction_id, query_time, query_mode])
                if 'torso' in file_name:
                    torso_question_set.append([question, answer, domain, question_type, static_or_dynamic, interaction_id, query_time, query_mode])
                if 'tail' in file_name:
                    tail_question_set.append([question, answer, domain, question_type, static_or_dynamic, interaction_id, query_time, query_mode])
    return head_question_set, torso_question_set, tail_question_set

def llama_answer(
    dialog, generator, tokenizer,
    temperature = 0,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    dialogs: List[Dialog] = [
        [ {"role": "system", "content": dialog['system']},
            {"role": "user", "content": dialog['user']}]]  
    prompt = tokenizer.apply_chat_template(
            dialogs,
            tokenize=False,
            add_generation_prompt=True,
        )
    response = generator(prompt, do_sample=False, eos_token_id=terminators)[0][0]['generated_text']
    output = response[len(prompt):].strip()

    return output

###################################################


def kb_qa(question, kb_content, generator, tokenizer, query_time):
    dialog = {}
    dialog['system'] = 'Please provide a brief answer as short as possible to the question based on your own knowledge and the following relevant CONTENT extracted from Knowledge Base. Answer "I don\'t know" if you are not confident of your answer. Please think step by step.'
    dialog['user'] = "The current query time is: " + query_time + '\n'
    dialog['user'] += question
    dialog['user'] += '\n'
    dialog['user'] += 'CONTENT: '
    if kb_content:
        dialog['user'] += str(kb_content)
    else:
        dialog['user'] += 'EMPTY'
    llama_current_answer = llama_answer(dialog, generator, temperature = 0, tokenizer=tokenizer).lower()
    print(dialog['user'])
    llama_current_answer = llama_current_answer.split("assistant<|end_header_id|>")[-1].strip()
    return llama_current_answer

def get_KB_results(kb_result_file=''):
    kb_result_file = '/kg_result/kg.json'
    id_to_kb_mapping = {}
    with open(kb_result_file, 'r') as file:
        data_list = json.load(file)
        for cur_item in data_list:
            id_to_kb_mapping[cur_item['interaction_id']] = cur_item['kg_response_str']
    return id_to_kb_mapping

def KERAG(json_file, question_set, generator, summarizer, tokenizer):
    id_to_kb_mapping = get_KB_results()
    kb_miss = 0
    kb_contain = 0
    total = 0
    with open(json_file, 'a', encoding='utf-8') as file:
        for cur_question, answer, domain, _, _, interaction_id, cur_time, _ in tqdm(question_set):
            query_time = cur_time
            kb_content = id_to_kb_mapping[interaction_id]
            total += 1
            if domain == 'finance':
                updated_kb_content = []
                kb_content = kb_content.split('<DOC>\n')
                cur_sys_prompt = generate_tool_prompt(finance_tools)
                cur_dialog = {}
                cur_dialog['system'] = cur_sys_prompt
                cur_dialog['user'] = "Question: "+cur_question
                llama_current_answer = llama_answer(cur_dialog, generator, temperature = 0, tokenizer=tokenizer).lower()
                llama_current_answer = llama_current_answer.split("assistant<|end_header_id|>")[-1].strip()
                print(llama_current_answer)
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
            
            if not kb_content:
                kb_miss += 1
            print(cur_question, answer)
            if str(answer).lower() in str(kb_content).lower():
                kb_contain += 1
                print('problem, KB hit', domain)
            else:
                print('problem, KB miss', domain)
            try:
                llm_answer = kb_qa(cur_question, kb_content, summarizer, tokenizer, query_time)
            except:
                flag = True
                tic = time.time()
                while flag:
                    try:
                        kb_content = kb_content[:int(len(kb_content)/2)]
                        llm_answer = kb_qa(cur_question, kb_content, summarizer,  tokenizer, query_time)
                        flag = False
                    except:
                        flag = True
                        toc = time.time()
                        if toc-tic > 60:
                            llm_answer = "I don\'t know"
                            break
            print('llama response', llm_answer)
            print('current gt', answer)
            item = {}
            item['domain'] = domain
            item['question'] = cur_question
            item['answer'] = answer
            item['llm_answer'] = llm_answer
            json_line = json.dumps(item)
            file.write(json_line + '\n')
    print('kb miss rate', kb_miss/total, 'kb contain rate', kb_contain/total)

import bz2
import re
from datetime import datetime

from loguru import logger
from prompts.templates import IN_CONTEXT_EXAMPLES, INSTRUCTIONS
from tqdm.auto import tqdm

def get_system_message():
    """Returns the system message containing instructions and in context examples."""
    return INSTRUCTIONS + "\n" + IN_CONTEXT_EXAMPLES

def log_response(messages, response, output_directory="api_responses"):
    """Save the response from the API to a file."""
    os.makedirs(output_directory, exist_ok=True)
    file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S.json")
    file_path = os.path.join(output_directory, file_name)
    with open(file_path, "w") as f:
        json.dump({"messages": messages, "response": response}, f)


def parse_response(response: str):
    """
    Return a tuple of (explanation, score) from the response, 
    where score is 0 if the prediction is wrong, 1 if the prediction is correct.

    Need to handle
    Corner case 1:
        {"explanation": ...}
        Wait, no! I made a mistake. The prediction does not exactly match the ground truth. ...
        {...}

    Corner case 2:
        {"score": 0, "explanation": "The prediction does not contain item, nick "goose" bradshaw, that is in the ground truth."}
        return a tuple of (explanation, score)
    """
    matches = re.findall(r"{([^}]*)}", response)
    text = ""
    for match in matches:
        text = "{" + match + "}"
    try:
        score = -1
        # Pattern to match the score
        score_pattern = r'"score"\s*:\s*(\d+)'
        score_match = re.search(score_pattern, text)
        if score_match:
            score = int(score_match.group(1))
            if score != 0 and score != 1:
                raise Exception("bad score: " + response)
        else:
            return "Parse Err: Score not found", -1

        # Pattern to match the explanation
        explanation_pattern = r'"explanation"\s*:\s*"(.+)"'
        explanation_match = re.search(explanation_pattern, text)
        if explanation_match:
            explanation = explanation_match.group(1)
            return explanation, score
        else:
            return text, score
    except Exception as e:
        print(f"Parsing Error with resp: {response}")
        print(f"Error: {e}")
        return response, -1


def trim_predictions_to_max_token_length(prediction):
    """Trims prediction output to 2000 tokens using Llama2 tokenizer"""
    max_token_length = 2000
    tokenized_prediction = tokenizer.encode(prediction)
    trimmed_tokenized_prediction = tokenized_prediction[1 : max_token_length + 1]
    trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
    return trimmed_prediction

def get_eval_response(tokenizer, generation_pipe, messages):
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
    )

    response = generation_pipe(prompt,
                    do_sample=False,
                    eos_token_id=terminators)[0]["generated_text"]
    # optional: output schema

    try:
        answer = response[len(prompt):].strip()
    except IndexError:
        # If the model fails to generate an answer, return a default response.
        answer = "no response"
    return answer

def load_predictions(path):
    queries, ground_truths, predictions = [], [], []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            # 去除行末的换行符
            line = line.strip()
            if line:  # 确保行不为空
                json_data = json.loads(line)
                query = json_data["question"]
                ground_truth = json_data["answer"]
                prediction = json_data["llm_answer"]
                queries.append(query)
                ground_truths.append(ground_truth)
                predictions.append(prediction)
    return queries, ground_truths, predictions

def evaluate_predictions(queries, ground_truths_list, predictions, tokenizer, generation_pipe, eval_result_json):
    """
    Evaluates the predictions generated by a model against ground truth answers.
    
    Args:
    queries (List[str]): List of queries.
    ground_truths_list (List[List[str]]): List of lists of ground truth answers. 
        Note each query can have multiple ground truth answers.
    predictions (list): List of predictions generated by the model.
    evaluation_model_name (str): Name of the evaluation model.
    
    Returns:
    dict: A dictionary containing evaluation results.
    """

    n_miss, n_correct = 0, 0
    system_message = get_system_message()
    with open(eval_result_json, 'a', encoding='utf-8') as file:
        for _idx, prediction in enumerate(tqdm(
            predictions, total=len(predictions), desc="Evaluating Predictions"
        )):
            flag = -1
            query = queries[_idx]
            ground_truth = str(ground_truths_list[_idx]).strip()
            #print(query, ground_truth, _idx)

            prediction = trim_predictions_to_max_token_length(prediction)
            prediction = prediction.strip()

            accuracy = -1

            ground_truth_lowercase = ground_truth.lower()
            prediction_lowercase = prediction.lower()
            prediction_lowercase_brief = prediction_lowercase.split("\n")[-1]
            messages = [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n",
                },
            ]
            if prediction_lowercase == ground_truth_lowercase or ground_truth_lowercase in prediction_lowercase or (str(ground_truth_lowercase).split("(")[0].strip().lower() in prediction_lowercase) or (ground_truth_lowercase in prediction_lowercase.replace('_', " ")):
                # exact correct
                accuracy = 1
                flag = 1
            elif 'don\'t know' in prediction_lowercase:
                if 'no' in ground_truth_lowercase or 'i don\'t know' in ground_truth_lowercase:
                    accuracy = 1
                    flag = 1
                else:
                    n_miss += 1
                    flag = 0

            elif "invalid" in prediction_lowercase and "invalid" in ground_truth_lowercase:
                accuracy = 1
                flag = 1
            elif "invalid" in prediction_lowercase and "invalid" not in ground_truth_lowercase:
                # hallucination
                accuracy = 0
                flag = -1
            elif "invalid" not in prediction_lowercase and "invalid" in ground_truth_lowercase:
                # hallucination
                accuracy = 0
                flag = -1
            else:
                response = get_eval_response(tokenizer, generation_pipe, messages)
                if response:
                    #log_response(messages, response)
                    _, accuracy = parse_response(response)
                    if not accuracy == 1:
                        messages_brief = [
                                {"role": "system", "content": system_message},
                                {
                                    "role": "user",
                                    "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction_lowercase_brief}\n",
                                },
                            ]
                        response_brief = get_eval_response(tokenizer, generation_pipe, messages_brief)
                        _, accuracy = parse_response(response_brief)

            if accuracy == 1:
                n_correct += 1
                flag = 1
            item = {}
            item["query"] = query
            item["ground_truth"] = ground_truth
            item["answer"] = prediction
            item["eval"] = flag
            print(item)
            json_line = json.dumps(item)
            file.write(json_line + '\n')

    n = len(predictions)
    results = {
        "score": (2 * n_correct + n_miss) / n - 1,
        "accuracy": n_correct / n,
        "hallucination": (n - n_correct - n_miss) / n,
        "missing": n_miss / n,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_hallucination": n - n_correct - n_miss,
        "total": n,
    }
    logger.info(results)
    return results

if __name__ == "__main__":
    crag_dir = '/data/CRAG/filtered_full_jsonl/test/'
    head_question_set, torso_question_set, tail_question_set = get_crag_questions_list(crag_dir)

    cur_model_dir = "/tuned/CRAG/summarizer-8B/"
    tokenizer = AutoTokenizer.from_pretrained(cur_model_dir)
    cur_model = AutoModelForCausalLM.from_pretrained(
    cur_model_dir, device_map="auto", attn_implementation = "flash_attention_2",
    torch_dtype = getattr(torch, "bfloat16")
    )
    print(cur_model.device)
    summarizer = pipeline(
        task="text-generation",
        model=cur_model,
        torch_dtype=torch.bfloat16,
        tokenizer=tokenizer,
        max_new_tokens=4096,
        device_map='auto'
    )

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
    all_eval_results = []

    print('Checking CRAG head questions')
    result_json_file = "/result/CRAG/KERAG-head.json" 
    KERAG(result_json_file, head_question_set, generation_pipe, summarizer, tokenizer)
    queries, ground_truths, predictions = load_predictions(result_json_file)
    evaluation_results = evaluate_predictions(
        queries, ground_truths, predictions, tokenizer, generation_pipe, "/result/CRAG/KERAG-head-evaled.json"
    )
    print(evaluation_results)
    all_eval_results.append(evaluation_results)
    
    print('Checking CRAG torso questions')
    result_json_file = "/result/CRAG/KERAG-torso.json" 
    KERAG(result_json_file, torso_question_set, generation_pipe, summarizer, tokenizer)
    queries, ground_truths, predictions = load_predictions(result_json_file)
    evaluation_results = evaluate_predictions(
        queries, ground_truths, predictions, tokenizer, generation_pipe, "/result/CRAG/KERAG-torso-evaled.json"
    )
    print(evaluation_results)
    all_eval_results.append(evaluation_results)
    
    print('Checking CRAG tail questions')
    result_json_file = "/result/CRAG/KERAG-tail.json" 
    KERAG(result_json_file, tail_question_set, generation_pipe, summarizer, tokenizer)
    queries, ground_truths, predictions = load_predictions(result_json_file)
    evaluation_results = evaluate_predictions(
        queries, ground_truths, predictions, tokenizer, generation_pipe, "/result/CRAG/KERAG-tail-evaled.json"
    )
    print(evaluation_results)
    all_eval_results.append(evaluation_results)

    total_score = 0
    total_acc = 0
    total_hal = 0
    total_miss = 0
    total_count = 0
    for cur_eval in all_eval_results:
        json_object = cur_eval
        total_score += json_object["score"]*json_object["total"]
        total_acc += json_object["accuracy"]*json_object["total"]
        total_hal += json_object["hallucination"]*json_object["total"]
        total_miss += json_object["missing"]*json_object["total"]
        total_count += json_object["total"]

    print(total_score/total_count)
    print(total_acc/total_count)
    print(total_hal/total_count)
    print(total_miss/total_count)
