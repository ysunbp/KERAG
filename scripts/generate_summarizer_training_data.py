import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3'
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
        if not '.jsonl' in file_name:
            continue
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
    
                if 'simple' in question_type: # use complex question finetuned summarizer does not perform as well to generate full dataset, use simple to generate simple dataset.
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
    #tokenizer = AutoTokenizer.from_pretrained(cur_model_dir)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    dialogs: List[Dialog] = [
        [ {"role": "system", "content": dialog['system']},
            {"role": "user", "content": dialog['user']}]]  
    #print(dialogs)
    prompt = tokenizer.apply_chat_template(
            dialogs,
            tokenize=False,
            add_generation_prompt=True,
        )
    #print(prompt)
    response = generator(prompt, do_sample=False, eos_token_id=terminators)[0][0]['generated_text']
    #print(response)
    output = response[len(prompt):].strip()
    #print(output)
    return output

###################################################

def compose_example(cur_tuple, gt):
    return '[EXAMPLE]: QUESTION: ' + cur_tuple[0] + '\n' + 'GROUND_TRUTH: ' + cur_tuple[2] + '\n' + 'ANSWER: ' + cur_tuple[1] + '\n' + 'Your answer: '+ gt + '\n'

def compose_eval_template(question, ground_truth, answer):
    global IN_CONTEXT_TRUE, IN_CONTEXT_FALSE
    dialog = {}
    example = 'Here are some examples:' + '\n'
    few_shot = ""
    for i in range(len(IN_CONTEXT_TRUE)):
        few_shot += compose_example(IN_CONTEXT_TRUE[i], "Yes")
        few_shot += compose_example(IN_CONTEXT_FALSE[i], "No")
    dialog['system'] = 'The task is provided a QUESTION with GROUND_TRUTH answer, evaluate whether my ANSWER is correct, answer briefly with Yes/No. You will first see some [EXAMPLE]s on this task and then you will complete the [TASK].'
    dialog['user'] = example+few_shot+'[TASK]: QUESTION: ' + str(question) + '\n' + 'GROUND_TRUTH: ' + str(ground_truth) + '\n' + 'ANSWER: ' + str(answer) + '\n' + "Your answer is?"
    return dialog

def kb_qa(question, kb_content, generator, tokenizer, query_time):
    dialog = {}
    dialog['system'] = 'Please provide a brief answer as short as possible to the question based on your own knowledge and the following relevant CONTENT extracted from Knowledge Base. Answer "I don\'t know" if you are not confident of your answer.'
    dialog['user'] = "The current query time is: " + query_time + '\n'
    dialog['user'] += question
    dialog['user'] += '\n'
    dialog['user'] += 'CONTENT: '
    if kb_content:
        dialog['user'] += str(kb_content)
    else:
        dialog['user'] += 'EMPTY'
    dialog['user'] += '\n'
    dialog['user'] += "Please think step by step."
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

INSTRUCTIONS = """Assume you are a human expert in grading predictions given by a model. You are given a question and a model prediction. Judge if the prediction matches the ground truth answer by following these steps:
1: Take it as granted that the Ground Truth is always correct.
2: If the Prediction indicates it is not sure about the answer, "score" should be "0"; otherwise, go the next step.
3: If the Prediction exactly matches the Ground Truth, "score" is 1.
4: If the Prediction does not exactly match the Ground Truth, go through the following steps and likely give a score as 0.
5: If the Ground Truth is a number, "score" is 1 if and only if the Prediction gives a number that almost exactly matches the ground truth.
6: If the Prediction is self-contradictory, "score" must be 0.
7: If the prediction is not answering the question, "score" must be 0.
8: If the prediction is a concise and correct summary of the ground truth, "score" is 1.
9: If ground truth contains a set of items, prediction must contain exactly same items for the score to be 1.
10: Otherwise, "score" is 0.

### Output a JSON blob with an "explanation" field explaining your answer as short as possible and an "score" field with value 1 or 0."""

IN_CONTEXT_EXAMPLES = """You should make the judgment based on provided examples.
Examples:
Question: "which company has higher eps, btu or cma?"
Ground Truth: "cma"
Prediction: "it is not possible to determine which company has a higher eps."
Output: {"score": 0, "explanation": "The prediction is not sure about the answer."}

Question: "who is taller, a or b?"
Ground Truth: "a"
Prediction: "The answer is a. a is of height 1.75 m and b is of height 1.82 m. So b is taller."
Output: {"score": 0, "explanation": "The prediction is self-contradictory."}

Question: "who among patricia place and lana clarkson is younger?"
Ground Truth: "lana clarkson"
Prediction: "lana clarkson (born april 5, 1962) is younger than patricia clarkson (born december 29, 1959)."
Output: {"score": 1, "explanation": "The prediction gives the correct answer."}

Question: "did lakers win more games than warriors in 2024-4?"
Ground Truth: "yes"
Prediction: "yes, lakers won more games than warriors in 2023-2024 season."
Output: {"score": 0, "explanation": "The prediction is not answering the question."}

Question: "what is the height of the building where citigroup is headquartered?"
Ground Truth: "151 m"
Prediction: "the height is 915 feet (279 m)."
Output: {"score": 0, "explanation": "The prediction, 151 m, does not match the ground truth, 279 m."}

Question: "who were the last 3 prime ministers of uk?"
Ground Truth: "rishi sunak, liz truss, and boris johnson"
Prediction: "1. liz truss, 2. boris johnson, 3.theresa may"
Output: {"score": 0, "explanation": "The prediction does not contain item, rishi sunak, that is in the grount truth."}

Question: "who authored the taming of the shrew?"
Ground Truth: "william shakespeare"
Prediction: "william shakespeare"
Output: {"score": 1, "explanation": "The prediction exactly matches the ground truth."}

Question: "who authored The Taming of the Shrew?"
Ground Truth: "william shakespeare"
Prediction: "w shakespeare"
Output: {"score": 1, "explanation": "The prediction matches the ground truth as w is the abbreviation of william."}

Question: "what is the state bird of california?"
Ground Truth: "california quail"
Prediction: "california valley quail"
Output: {"score": 1, "explanation": "The prediction matches the ground truth as california quail is also known as california valley quail."}

Question: "what country is maharashtra metro rail corporation limited located in?"
Ground truth: "india"
Prediction: "indianapolis"
Output: {"score": 0, "explanation": "The prediction doesn't match the ground truth answers."}

Question: "what was wolves's record in their last game in eng-premier league?"
Ground Truth: "w"
Prediction: "wolves's record in their last game in the eng-premier League was a 2-1 win over fulham."
Output: {"score": 1, "explanation": "The prediction correctly answers the result is a win."}

Question: "how many seconds is 3 minutes 15 seconds?"
Ground Truth: "195 seconds"
Prediction: "3 minutes 15 seconds is 195 seconds."
Output: {"score": 1, "explanation": "The prediction matches ground truth as it correctly states the number of seconds asked in the question."}

Question: "on which days did am distribute dividends in the last year?"
Ground Truth: "2023-01-24, 2023-04-25, 2023-07-25, 2023-10-24"
Prediction: "the company distributed dividends on october 24, 2023 and january 24, 2023."
Output: {"score": 0, "explanation": "The prediction partially matches the ground truth, but it is incomplete."}

Question: "on which days did xxx distribute dividends in the last year?"
Ground Truth: "2023-01-13, 2023-03-25, 2023-11-21"
Prediction: "xxx distributed dividends on 1. 2023-01-13, 2. 2023-03-25, 3. 2023-10-21."
Output: {"score": 0, "explanation": "The prediction contains an item, 2023-10-21, which is not in ground truth."}

Question: "who are current band members of maroon 5?"
Ground Truth: "adam Levine, jesse carmichael, james valentine, matt flynn, pj morton, sam farrar"
Prediction: "they are jesse carmichael, matt flynn, adam levine, pj morton, sam farrar, and james valentine."
Output: {"score": 1, "explanation": "The prediction exactly matches the ground truth."}

Question: "which movies comprise the matrix franchise?"
Ground Truth: "the matrix, the matrix reloaded, the matrix revolutions, the matrix resurrections"
Prediction: "the matrix, the matrix reloaded, the matrix revolutions, the animatrix, and the matrix resurrections."
Output: {"score": 0, "explanation": "The prediction covers more items than what are given by the ground truth."}

Question: "how deep is the deepest lake of new york?"
Ground Truth: "618 ft"
Prediction: "the deepest lake in new york is seneca lake, with a depth of 618.23 feet."
Output: {"score": 1, "explanation": "The prediction exactly matches the number in ground truth after rounding."}

Question: "what is the closing price of meta yesterday?"
Ground Truth: "$310.17"
Prediction: "310.2"
Output: {"score": 1, "explanation": "The prediction exactly matches the number in ground truth after rounding."}

Question: "what is the current market cap of appl?"
Ground Truth: "2.81 trillion"
Prediction: "2.667 trillion"
Output: {"score": 0, "explanation": "The prediction does not match the number in ground truth."}

Question: "what is the current pe ratio of appl?"
Ground Truth: "28.3"
Prediction: "the current pe ratio of apple is 26.66"
Output: {"score": 0, "explanation": "The prediction does not match the number in ground truth."}

Question: "how much is tesla's stock price down from its all-time high?"
Ground Truth: "$221.83"
Prediction: "209.52"
Output: {"score": 0, "explanation": "The prediction does not match the number in ground truth."}

Question: "what is the length of amazon river?"
Ground Truth: "over 4000 miles"
Prediction: "the length of amazon river is 4,000 miles"
Output: {"score": 0, "explanation": "The prediction does not say Amazon River is longer than 4000 miles."}

Question: "how many copies x were sold?"
Ground Truth: "2 million."
Prediction: "it is over 2 million"
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "what is the population of country x?"
Ground Truth: "3,576,873"
Prediction: "the population of country x is 3.3 million."
Output: {"score": 0, "explanation": "The prediction, 3.3 M, does not match the number, 3.6 M, in ground truth."}

Question: "what is the current market value of stock x?"
Ground Truth: "$2,237,578,268"
Prediction: "$2.16 billion."
Output: {"score": 0, "explanation": "The prediction, 2.16 B, does not match the number, 2.2 B, in ground truth."}

Question: "how old is michael jordan?"
Ground Truth: "61"
Prediction: "mj was born on February 17, 1963."
Output: {"score": 0, "explanation": "The prediction does not explicitly say the age."}

Question: "how long did amy teach as a teacher?"
Ground Truth: "24 years and 255 days"
Prediction: "25 years."
Output: {"score": 0, "explanation": "The prediction does not exactly match the ground truth."}

Question: "at what age did usher perform in the half time show of superbowl?"
Ground Truth: "45 years old"
Prediction: "usher has not performed at a super bowl halftime show."
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "what year did olivia rodrigo win the grammys best new artist award?"
Ground Truth: "2022"
Prediction: "sure! here's an answer in three words or fewer: 2022 grammy awards."
Output: {"score": 1, "explanation": "The prediction answers the correct number of the year."}

Question: "how many times have zendaya and tom holland starred in a film together?"
Ground Truth: "3"
Prediction: "they have starred in 3 films together: spider-man: homecoming, spider-man: far from home, and spider-man: no way home."
Output: {"score": 1, "explanation": "The prediction gives the right number and it is reasonable."}

Question: "who is younger, timothee chalamet or tom holland?"
Ground Truth: "tom holland"
Prediction: "timothée chalamet is younger than tom holland."
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "who had more number one hits on the us billboard, a or b?"
Ground Truth: "a had more number one hits on the us billboard than b, with 20 number one hits compared to b's 15."
Prediction: "a"
Output: {"score": 1, "explanation": "The prediction is a concise and correct summary of the ground truth."}

Question: "what is xxx's birthdate?"
Ground Truth: "1996-01-01."
Prediction: "02/01/1996"
Output: {"score": 0, "explanation": "The prediction does not match the ground truth."}

Question: "what was the worldwide box office haul for movie x?"
Ground Truth: "101756123."
Prediction: "102 million"
Output: {"score": 1, "explanation": "The prediction exactly matches the number in ground truth after rounding."}

Question: "how much has spotify's user base increased by since 2020 in na?"
Ground Truth: "spotify's user base increased by 34 million since 2020."
Prediction: "spotify's north american user base increased from 36 million in 2020 to 85 million by 2021"
Output: {"score": 0, "explanation": "The prediction is not answering the question as it only gives the increase from 2020 to 2021."}
"""

import bz2
import re
from datetime import datetime

from loguru import logger
from tqdm.auto import tqdm

def load_json_file(file_path):
    """Load and return the content of a JSON file."""
    logger.info(f"Loading JSON from {file_path}")
    with open(file_path) as f:
        return json.load(f)


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


def load_data_in_batches(dataset_path, batch_size):
    """
    Generator function that reads data from a compressed file and yields batches of data.
    Each batch is a dictionary containing lists of interaction_ids, queries, search results, query times, and answers.
    
    Args:
    dataset_path (str): Path to the dataset file.
    batch_size (int): Number of data items in each batch.
    
    Yields:
    dict: A batch of data.
    """
    def initialize_batch():
        """ Helper function to create an empty batch. """
        return {"interaction_id": [], "query": [], "search_results": [], "query_time": [], "answer": []}

    try:
        with bz2.open(dataset_path, "rt") as file:
            batch = initialize_batch()
            for line in file:
                try:
                    item = json.loads(line)
                    for key in batch:
                        batch[key].append(item[key])
                    
                    if len(batch["query"]) == batch_size:
                        yield batch
                        batch = initialize_batch()
                except json.JSONDecodeError:
                    logger.warn("Warning: Failed to decode a line.")
            # Yield any remaining data as the last batch
            if batch["query"]:
                yield batch
    except FileNotFoundError as e:
        logger.error(f"Error: The file {dataset_path} was not found.")
        raise e
    except IOError as e:
        logger.error(f"Error: An error occurred while reading the file {dataset_path}.")
        raise e

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

def evaluate_predictions(queries, ground_truths_list, predictions, tokenizer, generation_pipe):
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

    for _idx, prediction in enumerate(tqdm(
        predictions, total=len(predictions), desc="Evaluating Predictions"
    )):
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
        elif 'don\'t know' in prediction_lowercase:
            if 'no' in ground_truth_lowercase or 'i don\'t know' in ground_truth_lowercase:
                accuracy = 1
            else:
                n_miss += 1

        elif "invalid" in prediction_lowercase and "invalid" in ground_truth_lowercase:
            accuracy = 1
        elif "invalid" in prediction_lowercase and "invalid" not in ground_truth_lowercase:
            # hallucination
            accuracy = 0
        elif "invalid" not in prediction_lowercase and "invalid" in ground_truth_lowercase:
            # hallucination
            accuracy = 0
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

def populate_ground_truth(json_file, question_set, generator, summarizer, tokenizer):
    id_to_kb_mapping = get_KB_results()
    with open(json_file, 'a', encoding='utf-8') as file:
        for cur_question, answer, domain, cur_question_t, static_or_dynamic, interaction_id, cur_time, query_mode_ab in tqdm(question_set):
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
                cur_training_gt = 'I don\'t know.'
                cur_mode = 'B'
            else:
                #print(cur_question, answer)
                if str(answer).lower() in str(kb_content).lower():
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
                if (str(answer).lower() in llm_answer) or (str(answer).split("(")[0].strip().lower() in llm_answer) or (str(answer).lower() in llm_answer.replace('_', " ")):
                    cur_training_gt = llm_answer
                    cur_mode = 'B'
                else:
                    llm_answer_brief = llm_answer.split("\n")[-1]
                    print('Auto eval processed answer', llm_answer_brief)
                    system_message = get_system_message()
                    messages = [
                        {"role": "system", "content": system_message},
                        {
                            "role": "user",
                            "content": f"Question: {cur_question}\n Ground truth: {answer}\n Prediction: {llm_answer}\n",
                        },
                    ]
                    response = get_eval_response(tokenizer, generation_pipe, messages)
                    if response:
                        #log_response(messages, response)
                        _, accuracy = parse_response(response)
                        if not accuracy == 1:
                            messages_brief = [
                                    {"role": "system", "content": system_message},
                                    {
                                        "role": "user",
                                        "content": f"Question: {cur_question}\n Ground truth: {answer}\n Prediction: {llm_answer_brief}\n",
                                    },
                                ]
                            response_brief = get_eval_response(tokenizer, generation_pipe, messages_brief)
                            _, accuracy = parse_response(response_brief)

                    if accuracy == 1:
                        cur_training_gt = llm_answer
                        cur_mode = 'B'
                    else:
                        cur_training_gt = answer.lower()
                        cur_mode = 'A'
                    
            item = {}
            item["query"] = cur_question
            item["answer"] = cur_training_gt
            item["question_type"] = cur_question_t
            item["domain"] = domain
            item["query_time"] = cur_time
            item["static_or_dynamic"] = static_or_dynamic
            item["interaction_id"] = interaction_id
            item["query_mode"] = cur_mode
            json_line = json.dumps(item)
            file.write(json_line + '\n')

if __name__ == "__main__":
    crag_dir = '/data/CRAG/filtered_full_jsonl/valid/'
    head_question_set, torso_question_set, tail_question_set = get_crag_questions_list(crag_dir)

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
    
    summarizer = generation_pipe
    
    print('Checking CRAG head questions')
    result_json_file = "/data/CRAG/filtered_full_jsonl/valid/simple_dataset/head-populated-updated.json"
    populate_ground_truth(result_json_file, head_question_set, generation_pipe, summarizer, tokenizer)

    
    print('Checking CRAG torso questions')
    result_json_file = "/data/CRAG/filtered_full_jsonl/valid/simple_dataset/torso-populated-updated.json"
    populate_ground_truth(result_json_file, torso_question_set, generation_pipe, summarizer, tokenizer)
    
    print('Checking CRAG tail questions')
    result_json_file = "/data/CRAG/filtered_full_jsonl/valid/simple_dataset/tail-populated-updated.json"
    populate_ground_truth(result_json_file, tail_question_set, generation_pipe, summarizer, tokenizer)

