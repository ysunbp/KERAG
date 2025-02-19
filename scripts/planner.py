import os
import sys
from typing import List
sys.path.append('crag-mock-api/apiwrapper')
from pycragapi import CRAG
import json
import torch
from transformers import (
    AutoTokenizer,
    pipeline,
)
from planner_cot_template import ZERO_SHOT_PLANNER_TEMPLATE, COT_TEMPLATE_FOR_DATE
from json import JSONDecoder
import os
from tqdm.auto import tqdm
import re
from datetime import datetime
from utils import trim_predictions_to_token_length
from dateutil import parser
from tqdm import tqdm
import csv
from parse_time_period import parse_date_range

def get_ticker_name_list(file_path='./crag-mock-api/cragkg/finance/company_name.dict'):
    # 读取 CSV 文件
    ticker_list = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        # 跳过表头（如果有）
        next(csv_reader)
        for row in csv_reader:
            ticker_list.append(row[-1])
    return ticker_list

def process_team_name(team_name):
    return team_name.title()

def parse_date(date_string, query_time):
    try:
        # 检查是否仅为年份
        if len(date_string) == 4 and date_string.isdigit():
            return date_string  # 返回 YYYY 格式
        
        # 检查是否为 YYYY-MM 格式
        if len(date_string) == 7 and date_string[4] == '-' and date_string[:4].isdigit() and date_string[5:7].isdigit():
            return date_string  # 返回 YYYY-MM 格式

        # 解析日期字符串
        parsed_date = parser.parse(date_string)
        # 格式化为统一格式
        return parsed_date.strftime('%Y-%m-%d')
    except ValueError:
        parsed_date = parser.parse(query_time)
        return parsed_date.strftime('%Y-%m-%d')
    
def get_sports_db_to_json(date, teams1, teams2, points1, points2):
    all_games = []
    for i in range(len(date)):
        cur_game = {}
        cur_game['date'] = date[i]
        cur_game['teams'] = teams1[i] + ' vs ' + teams2[i]
        cur_game['result'] = str(points1[i]) + ' - ' + str(points2[i])
        all_games.append(cur_game)
    return all_games

def get_sports_soccer_db_to_json(date, teams, points1, points2, host_team_captains):
    all_games = []
    for i in range(len(date)):
        cur_game = {}
        cur_game['date'] = date[i]
        cur_game['teams'] = teams[i]
        cur_game['result'] = str(points1[i]) + ' - ' + str(points2[i])
        cur_game['team captain'] = host_team_captains[i]
        all_games.append(cur_game)
    return all_games
#uvicorn server:app --reload

def llama_answer(
    dialog, generator, tokenizer,
    temperature = 0,
    top_p = 0.9,
    max_gen_len = None
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
    dialogs = [
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


planner_res_path = '/kg_result/str.json'
kg_res_path = '/kg_result/kg.json'
model_path =  "meta-llama/Llama-3.1-70B-Instruct"


import sqlite3 as sql
import pandas as pd
KG_BASE_DIRECTORY = './crag-mock-api/cragkg/'
nba_kg_file = os.path.join(KG_BASE_DIRECTORY, "sports", 'nba.sqlite')
conn = sql.connect(nba_kg_file) # create connection object to database
df_game_by_team_home = pd.read_sql(f"select distinct team_name_home from game", conn).values.tolist()
df_game_by_team_away = pd.read_sql(f"select distinct team_name_away from game", conn).values.tolist()
nba_names = []
for item in df_game_by_team_home:
    if not item[0] in nba_names:
        nba_names.append(item[0])
for item in df_game_by_team_away:
    if not item[0] in nba_names:
        nba_names.append(item[0])
file_name = 'soccer_team_match_stats.pkl'
soccer_kg_file = os.path.join(KG_BASE_DIRECTORY, "sports", file_name)
team_match_stats = pd.read_pickle(os.path.join(KG_BASE_DIRECTORY, "sports", file_name))
team_match_stats = team_match_stats[team_match_stats.index.get_level_values('league').notna()]
soccer_names = list(team_match_stats['GF'].reset_index()['team'].unique())
#print(soccer_names)

from rank_bm25 import BM25Okapi

def find_most_similar_item(string, item_list):
    # 将字符串和列表项分词
    tokenized_corpus = [item.split() for item in item_list]
    tokenized_query = string.split()

    # 初始化 BM25
    bm25 = BM25Okapi(tokenized_corpus)

    # 计算相似度
    scores = bm25.get_scores(tokenized_query)

    # 找到最高分的索引
    most_similar_index = scores.argmax()

    return item_list[most_similar_index]

def extract_json_objects(text, decoder=JSONDecoder()):
    """Find JSON objects in text, and yield the decoded JSON data

    Does not attempt to look for JSON arrays, text, or other JSON types outside
    of a parent JSON object.

    """
    pos = 0
    results = []
    while True:
        match = text.find("{", pos)
        # print("match: ", match)
        if match == -1:
            break
        try:
            # print("text[match:]: ", text[match:])
            result, index = decoder.raw_decode(text[match:])
            results.append(result)
            # yield result
            pos = match + index
        except ValueError:
            pos = match + 1
    return results


def flatten_json(maybe_nested_json):
    flattened_json = {}
    is_flatten_json = True
    for k, v in maybe_nested_json.items():
        if isinstance(v, dict):
            is_flatten_json = False
            flattened_nested, _ = flatten_json(v)
            # merge
            flattened_json = {**flattened_json, **flattened_nested}
        else:
            flattened_json[k] = v
    return flattened_json, is_flatten_json



if not os.path.exists(planner_res_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    generation_pipe = pipeline(
        task="text-generation",
        model=model_path,
        torch_dtype=torch.bfloat16,
        tokenizer=tokenizer,
        max_new_tokens=4096,
        device_map='auto'
    )
    
    v3_data_test_split = '/data/CRAG/filtered_full_jsonl/test/'
    
    def read_jsonl(file_name):
        data = []
        with open(file_name, "r") as f:
            for _, line in enumerate(f):
                data.append(json.loads(line))
        return data
    file_names = [
        os.path.join(v3_data_test_split, i) for i in os.listdir(v3_data_test_split)
    ]
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        test_data = []
        for i in file_names:
            test_data += read_jsonl(i)

    planner_responses = []
    for pred in tqdm(test_data):
        query = pred['query']
        query_time = pred['query_time']

        messages = [
            {
                "role": "system",
                "content": ZERO_SHOT_PLANNER_TEMPLATE
            },
            {
                "role": "user",
                "content": """
    ### Query
    {query}
    """.format(query = query)
            },
        ]

        date_messages = [
            {
                "role": "system",
                "content": COT_TEMPLATE_FOR_DATE
            },
            {
                "role": "user",
                "content": """
    ### Query
    {query}
    ### Query Time
    {query_time}
    """.format(query = query, query_time = query_time)
            },
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        response = generation_pipe(prompt,
                                   do_sample=False, 
                                   eos_token_id=terminators)[0]["generated_text"]
        output = response[len(prompt) :].strip()

        date_prompt = tokenizer.apply_chat_template(
            date_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        date_response = generation_pipe(date_prompt,
                                   do_sample=False, 
                                   eos_token_id=terminators)[0]["generated_text"]
        date_output = date_response[len(date_prompt) :].strip()
        extracted_date = date_output.split('<')[-1].split('>')[0]

        # breakpoint()
        try:
            res = json.loads(output)
        except Exception:
            res = extract_json_objects(output)
            if len(res) != 1:
                res = res[0]
        res['datetime'] = extracted_date
        planner_responses.append(
            {
                'interaction_id': pred['interaction_id'],
                'query': query,
                'query_time': query_time,
                'query_extract': flatten_json(res)[0]
            }
        )
    # breakpoint()
    with open(planner_res_path, 'w') as f:
        json.dump(planner_responses, f)

else:
    print('planner results exist')
    with open(planner_res_path, 'r') as f:
        planner_responses = json.load(f)

ticker_name_list = get_ticker_name_list()
api = CRAG()
res = []

total_count = 0
domain_hit_count = 0

open_count = 0
finance_count = 0
movie_count = 0
sport_count = 0
music_count = 0

open_entity_count = 0
finance_entity_count = 0
movie_entity_movie_count = 0
movie_entity_person_count = 0
movie_entity_year_count = 0
sport_entity_nba_count = 0
sport_entity_soccer_count = 0
music_entity_artist_count = 0
music_entity_song_count = 0

open_miss_ref_count = 0
finance_miss_ref_count = 0
movie_miss_ref_count = 0
sport_miss_ref_count = 0
music_miss_ref_count = 0



for response in tqdm(planner_responses):

    total_count += 1

    reference = []
    ref = ''
    # length = 0
    planner = response['query_extract']
    query = response['query']
    query_time = response['query_time']
    #if not query == 'on average, what was the daily high stock price of xpev over the past week?':
    #    continue
    
    if 'domain' in planner.keys():
        domain = planner['domain']
        domain_hit_count += 1
        if domain in ['encyclopedia', 'other']:
            open_count += 1
            if 'main_entity' in planner.keys():
                open_entity_count += 1
                potential_miss = False
                try:
                    top_entity_names = api.open_search_entity_by_name(planner['main_entity'])['result']
                    top_entity_names_lower = [item.lower() for item in top_entity_names]
                    if planner['main_entity'].lower() in top_entity_names_lower:
                        cur_idx = top_entity_names_lower.index(planner['main_entity'].lower())
                        top_entity_name = top_entity_names[cur_idx]
                    else:
                        top_entity_name = top_entity_names[0]
                        potential_miss = True
                    ref = api.open_get_entity(top_entity_name)['result']
                    reference.append({top_entity_name: ref})
                    #print(top_entity_name, ref)
                except:
                    try:
                        top_entity_names = api.open_search_entity_by_name(query)['result']
                        top_entity_name = top_entity_names[0]
                        ref = api.open_get_entity(top_entity_name)['result']
                        reference.append({top_entity_name: ref})
                        #print(top_entity_name, ref)
                    except:
                        potential_miss = True
        
            if len(reference) == 0 or potential_miss:
                open_miss_ref_count += 1
                print('open entity miss')
                try:
                    person = planner['main_entity']
                    ref = api.movie_get_person_info(person)['result'][0]
                    if 'acted_movies' in ref.keys():
                        current_acted = []
                        cur_film_ids = ref['acted_movies']
                        for cur_film_id in cur_film_ids:
                            try:
                                movie_info = api.movie_get_movie_info_by_id(int(cur_film_id))['result']
                                movie_name = movie_info['title']
                                movie_release = movie_info['release_date']
                                movie_box = movie_info['revenue']
                                if movie_box == 0:
                                    movie_box = 'no information'
                                current_info = movie_name + " (released on " + movie_release + ", box office: " +movie_box+ ")"
                                current_acted.append(current_info)
                            except:
                                print('no id', cur_film_id)
                                continue
                        ref['acted_movies'] = current_acted
                    if 'directed_movies' in ref.keys():
                        current_directed = []
                        cur_film_ids = ref['directed_movies']
                        for cur_film_id in cur_film_ids:
                            try:
                                movie_info = api.movie_get_movie_info_by_id(int(cur_film_id))['result']
                                movie_name = movie_info['title']
                                movie_release = movie_info['release_date']
                                movie_box = movie_info['revenue']
                                if movie_box == 0:
                                    movie_box = 'no information'
                                current_info = movie_name + " (released on " + movie_release + ", box office: " +movie_box+ ")"
                                current_directed.append(current_info)
                            except:
                                print('no id', cur_film_id)
                                continue
                        ref['directed_movies'] = current_directed
                    print('updated ref', ref)
                    reference.append({person + '_' + 'person information': ref})
                except:
                    print('not movie person')
                try:
                    top_artist_name = api.music_search_artist_entity_by_name(artist_name)['result'][0]
                    if not top_artist_name.lower() in query:
                        flag = True
                    ref = api.music_get_members(top_artist_name)['result']
                    reference.append({top_artist_name + '_' + 'member': ref})
                    ref = api.music_get_artist_birth_place(top_artist_name)['result']
                    reference.append({top_artist_name + '_' + 'birth place': ref})
                    ref = api.music_get_artist_birth_date(top_artist_name)['result']
                    reference.append({top_artist_name + '_' + 'birth date': ref})
                    ref = api.music_get_lifespan(top_artist_name)['result']
                    reference.append({top_artist_name + '_' + 'lifespan': ref})
                    all_songs = api.music_get_artist_all_works(top_artist_name)['result']
                    ref = []
                    for cur_song in all_songs:
                        try:
                            cur_song_date = api.music_get_song_release_date(cur_song)['result']
                            ref.append(cur_song + ' (released on '+cur_song_date+')')
                        except:
                            ref.append(cur_song)
                            continue
                    reference.append({top_artist_name + '_' + 'artist work': ref})
                    ref = api.music_grammy_get_award_count_by_artist(top_artist_name)['result']
                    reference.append({top_artist_name + '_' + 'grammy award count': ref})
                    ref = api.music_grammy_get_award_date_by_artist(top_artist_name)['result']
                    reference.append({top_artist_name + '_' + 'grammy award date': ref})
                    billboard_list = api.music_get_billboard_rank_date(1)[1]
                    billboard_list_lower = [item.lower() for item in billboard_list]
                    ref = billboard_list.count(artist_name)
                    reference.append({artist_name + '_' + 'billboard top-1 count': ref})
                except:
                    print('not music person')
                print(reference)
            print(query)
            print(planner)
            



        
        if domain == 'finance':
            finance_count += 1
            
            if 'market_identifier' in planner.keys():
                finance_entity_count += 1
                market_identifier = planner['market_identifier']
                #print(market_identifier)
                if isinstance(market_identifier, list):
                    market_identifier = market_identifier[0]
                if market_identifier.upper() in ticker_name_list:
                    ticker_name = market_identifier.upper()
                else:
                    try:
                        market_identifier = api.finance_get_company_name(market_identifier)['result'][0]
                        ticker_name = api.finance_get_ticker_by_name(market_identifier)['result']
                    except:
                        ticker_name = market_identifier.upper()

                date_pattern = r'(\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}[-/]\d{1,2}[-/]?\d{0,2}|\d{4})'
                if 'datetime' in planner.keys():
                    if "~" in planner['datetime']:
                        
                        period_one, period_two = planner['datetime'].split("~")
                        if "past" in period_one:
                            match_str = ["~"+period_two]
                        elif "past" in period_two:
                            match_str = ["~"+period_one]
                        elif "future" in period_one:
                            match_str = [period_two+"~"]
                        elif "future" in period_two:
                            match_str = [period_one+"~"]
                        else:
                            match_str = [period_one+"~~"+period_two]
                    else:
                        found_time = re.findall(date_pattern, planner['datetime'])
                        if found_time:
                            match_str = found_time
                        else:
                            match_str = [query_time]
                else:
                    match_str = []
                querytime = []
                for cur_date in match_str:
                    if cur_date[0] == "~":
                        date = parse_date(cur_date[1:], query_time)
                        if len(date) == 10:
                            querytime = [date + " 00:00:00 EST"]
                        else:
                            querytime = [date]
                    elif cur_date[-1] == "~":
                        date = parse_date(cur_date[:-1], query_time)
                        if len(date) == 10:
                            querytime = [date + " 00:00:00 EST"]
                        else:
                            querytime = [date]
                    elif "~~" in cur_date:
                        date1 = parse_date(cur_date.split("~~")[0], query_time)
                        if len(date1) == 10:
                            querytime1 = date1 + " 00:00:00 EST"
                        else:
                            querytime1 = date1
                        date2 = parse_date(cur_date.split("~~")[1], query_time)
                        if len(date2) == 10:
                            querytime2 = date2 + " 00:00:00 EST"
                        else:
                            querytime2 = date2
                        querytime = [querytime1, querytime2]
                    else:
                        date = parse_date(cur_date, query_time)
                        if len(date) == 10:
                            querytime = [date + " 00:00:00 EST"]
                        else:
                            querytime = [date]
                    print(querytime, planner, query)
                    ref = api.finance_get_price_history(ticker_name)['result']
                    if ref:
                        if len(querytime) == 1:
                            if querytime[0] in ref.keys():
                                print('get price history')
                                cur_ref = ref[querytime[0]]
                                reference.append({market_identifier + '_' + 'price ' + querytime[0] : cur_ref})
                        elif len(querytime) == 2:
                            time_points = parse_date_range(querytime)
                            #print("keys",ref.keys())
                            for cur_time_point in time_points:
                                #print(cur_time_point)
                                cur_time_point = cur_time_point  + " 00:00:00 EST"
                                #print(cur_time_point)
                                if cur_time_point in ref.keys():
                                    print('get price history')
                                    cur_ref = ref[cur_time_point]
                                    reference.append({market_identifier + '_' + 'price ' + cur_time_point : cur_ref})
                    ref = api.finance_get_detailed_price_history(ticker_name)['result']
                    if ref:
                        if len(querytime) == 1:
                            for cur_key in ref.keys():
                                if querytime[0][:10] == cur_key[:10]:
                                    print('get detailed price history')
                                    cur_price_ref = ref[cur_key]
                                    reference.append({market_identifier + '_' + 'detailed real-time changing price ' + cur_key: cur_price_ref})
                        elif len(querytime) == 2:
                            time_points = parse_date_range(querytime)
                            for cur_key in ref.keys():
                                for cur_time_point in time_points:
                                    if cur_time_point[:10] == cur_key[:10]:
                                        print('get detailed price history')
                                        cur_price_ref = ref[cur_key]
                                        reference.append({market_identifier + '_' + 'detailed real-time changing price ' + cur_key: cur_price_ref})
                    ref = api.finance_get_dividends_history(ticker_name)['result']
                    if ref:
                        print('get dividend history')
                        reference.append({market_identifier + '_' + 'dividend': ref})
                
                ref = api.finance_get_market_capitalization(ticker_name)['result']
                reference.append({market_identifier + '_' + 'marketCap': ref})

                ref = api.finance_get_eps(ticker_name)['result']
                reference.append({market_identifier + '_' + 'EPS': ref})

                ref = api.finance_get_pe_ratio(ticker_name)['result']
                reference.append({market_identifier + '_' + 'P/E ratio': ref})

                ref = api.finance_get_info(ticker_name)['result']
                reference.append({market_identifier + '_' + 'other': ref})

                

            if len(reference) == 0:
                finance_miss_ref_count += 1
                print(query)
                print(planner)
                print('finance entity miss')
            else:
                print(query)
                print(planner)
                print('finance entity hit')

        if domain == 'movie':
            movie_count += 1
            if 'movie_name' in planner.keys() and planner['movie_name'] is not None:
                movie_entity_movie_count += 1
                if isinstance(planner['movie_name'], str):
                    movie_names = planner['movie_name'].split(',')
                else:
                    movie_names = planner['movie_name']
                for movie_name in movie_names:
                    try:
                        ref = api.movie_get_movie_info(movie_name)['result'][0]
                        reference.append({movie_name + '_' + 'movie information': ref})
                    except:
                        pass
                #print(movie_names, reference)
                
            if 'person' in planner.keys() and planner['person'] is not None:
                movie_entity_person_count += 1
                if isinstance(planner['person'], str):
                    person_list = planner['person'].split(',')
                else:
                    person_list = planner['person']
                for person in person_list:
                    try:
                        ref = api.movie_get_person_info(person)['result'][0]
                        if 'acted_movies' in ref.keys():
                            current_acted = []
                            cur_film_ids = ref['acted_movies']
                            for cur_film_id in cur_film_ids:
                                try:
                                    movie_info = api.movie_get_movie_info_by_id(int(cur_film_id))['result']
                                    movie_name = movie_info['title']
                                    movie_release = movie_info['release_date']
                                    movie_box = movie_info['revenue']
                                    if movie_box == 0:
                                        movie_box = 'no information'
                                    current_info = movie_name + " (released on " + movie_release + ", box office: " +movie_box+ ")"
                                    current_acted.append(current_info)
                                except:
                                    print('no id', cur_film_id)
                                    continue
                            ref['acted_movies'] = current_acted
                        if 'directed_movies' in ref.keys():
                            current_directed = []
                            cur_film_ids = ref['directed_movies']
                            for cur_film_id in cur_film_ids:
                                try:
                                    movie_info = api.movie_get_movie_info_by_id(int(cur_film_id))['result']
                                    movie_name = movie_info['title']
                                    movie_release = movie_info['release_date']
                                    movie_box = movie_info['revenue']
                                    if movie_box == 0:
                                        movie_box = 'no information'
                                    current_info = movie_name + " (released on " + movie_release + ", box office: " +movie_box+ ")"
                                    current_directed.append(current_info)
                                except:
                                    print('no id', cur_film_id)
                                    continue
                            ref['directed_movies'] = current_directed
                        print('updated ref', ref)
                        reference.append({person + '_' + 'person information': ref})
                        if not ref['name'].lower() in query: # potentially wrong in matching domain
                            print('multi movie')
                            try:
                                top_entity_name = api.open_search_entity_by_name(planner['person'])['result'][0]
                                ref = api.open_get_entity(top_entity_name)['result']
                                reference.append({top_entity_name: ref})
                            except:
                                print('no open')
                            try:
                                artist_names = planner['person'].split(',') if isinstance(planner['person'], str) else planner['person']
                                for artist_name in artist_names:
                                    try:
                                        top_artist_name = api.music_search_artist_entity_by_name(artist_name)['result'][0]
                                        ref = api.music_get_members(top_artist_name)['result']
                                        reference.append({top_artist_name + '_' + 'member': ref})
                                        ref = api.music_get_artist_birth_place(top_artist_name)['result']
                                        reference.append({top_artist_name + '_' + 'birth place': ref})
                                        ref = api.music_get_artist_birth_date(top_artist_name)['result']
                                        reference.append({top_artist_name + '_' + 'birth date': ref})
                                        ref = api.music_get_lifespan(top_artist_name)['result']
                                        reference.append({top_artist_name + '_' + 'lifespan': ref})
                                        all_songs = api.music_get_artist_all_works(top_artist_name)['result']
                                        ref = []
                                        for cur_song in all_songs:
                                            try:
                                                cur_song_date = api.music_get_song_release_date(cur_song)['result']
                                                ref.append(cur_song + ' (released on '+cur_song_date+')')
                                            except:
                                                ref.append(cur_song)
                                                continue
                                        reference.append({top_artist_name + '_' + 'artist work': ref})
                                        ref = api.music_grammy_get_award_count_by_artist(top_artist_name)['result']
                                        reference.append({top_artist_name + '_' + 'grammy award count': ref})
                                        ref = api.music_grammy_get_award_date_by_artist(top_artist_name)['result']
                                        reference.append({top_artist_name + '_' + 'grammy award date': ref})
                                    except:
                                        continue
                            except:
                                print('no music')
                    except:
                        pass
            if 'year' in planner.keys() and planner['year'] is not None:
                movie_entity_year_count += 1
                if isinstance(planner['year'], str) or isinstance(planner['year'], int):
                    years = str(planner['year']).split(',')
                else:
                    years = planner['year']
                for year in years:
                    try:
                        ref = api.movie_get_year_info(year)['result']
                        all_movies = []
                        oscar_movies = {}
                        oscar_actors = {}
                        for movie_id in ref['movie_list']:
                            all_movies.append(api.movie_get_movie_info_by_id(movie_id)['result']['title'])
                        for movie_dict in ref['oscar_awards']: # debugged by Yushi
                            award_category = movie_dict['category']
                            if 'actor' in award_category.lower() or 'actress' in award_category.lower():
                                if movie_dict['winner']:
                                    award_key = award_category + '_' + 'winner'
                                else:
                                    award_key = award_category + '_' + 'nominated'
                                if not award_key in oscar_actors:
                                    oscar_actors[award_key] = [movie_dict['name']]
                                else:
                                    oscar_actors[award_key].append(movie_dict['name'])
                            else:
                                if movie_dict['winner']:
                                    oscar_movies[award_category] = movie_dict['film']
                        reference.append({year + '_oscar_movies': oscar_movies})
                        reference.append({year + '_oscar_actor' : oscar_actors})
                    except:
                        pass

            if len(reference) == 0:
                movie_miss_ref_count += 1
                print('movie entity miss')
            print(query)
            print(planner)


        if domain == 'sports':
            sport_count += 1
            date = 'empty'

            if 'sport_type' in planner.keys():
                if planner['sport_type'] == 'basketball':
                    sport_entity_nba_count += 1
                    date_pattern = r'(\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}[-/]\d{1,2}[-/]?\d{0,2}|\d{4})'

                    #if 'datetime' in planner.keys():
                    #    match_str = re.findall(date_pattern, planner['datetime'])
                    #else:
                    #    match_str = []
                    if 'datetime' in planner.keys():
                        if "~" in planner['datetime']:
                            period_one, period_two = planner['datetime'].split("~")
                            if "past" in period_one:
                                match_str = ["~"+period_two]
                            elif "past" in period_two:
                                match_str = ["~"+period_one]
                            elif "future" in period_one:
                                match_str = [period_two+"~"]
                            elif "future" in period_two:
                                match_str = [period_one+"~"]
                            else:
                                match_str = [period_one+"~~"+period_two]
                        else:
                            found_time = re.findall(date_pattern, planner['datetime'])
                            if found_time:
                                match_str = found_time
                            else:
                                match_str = [query_time]
                    else:
                        match_str = []
                    querytime = []
                    for cur_date in match_str: 
                        if 'team' in planner.keys():
                            if cur_date[0] == "~":
                                date = parse_date(cur_date[1:], query_time)
                                if len(date) == 10:
                                    querytime = [date + " 00:00:00 EST"]
                                else:
                                    querytime = [date]
                            elif cur_date[-1] == "~":
                                date = parse_date(cur_date[:-1], query_time)
                                if len(date) == 10:
                                    querytime = [date + " 00:00:00 EST"]
                                else:
                                    querytime = [date]
                            elif "~~" in cur_date:
                                date1 = parse_date(cur_date.split("~~")[0], query_time)
                                if len(date1) == 10:
                                    querytime1 = date1 + " 00:00:00 EST"
                                else:
                                    querytime1 = date1
                                date2 = parse_date(cur_date.split("~~")[1], query_time)
                                if len(date2) == 10:
                                    querytime2 = date2 + " 00:00:00 EST"
                                else:
                                    querytime2 = date2
                                querytime = [querytime1, querytime2]
                            else:
                                date = parse_date(cur_date, query_time)
                                if len(date) == 10:
                                    querytime = [date + " 00:00:00 EST"]
                                else:
                                    querytime = [date]
                            
                            print(querytime, planner, query)

                            try:
                                cur_team_name = find_most_similar_item(process_team_name(planner['team']), nba_names)
                                if len(querytime) == 1:
                                    ref = api.sports_nba_get_games_on_date(querytime[0][:4], cur_team_name)['result']
                                    if ref:
                                        match_date = list(ref['game_date'].values())
                                        team_home = list(ref['team_name_home'].values())
                                        team_away = list(ref['team_name_away'].values())
                                        pts_home = list(ref['pts_home'].values())
                                        pts_away = list(ref['pts_away'].values())
                                        result_nba = get_sports_db_to_json(match_date, team_home, team_away, pts_home, pts_away)
                                        print('current result', result_nba)
                                        reference.append({'nba games: ': result_nba})
                                elif len(querytime) == 2:
                                    time_points = parse_date_range(querytime)
                                    time_points_set = []
                                    for cur_time_point in time_points:
                                        if not cur_time_point[:4] in time_points_set:
                                            time_points_set.append(cur_time_point[:4])
                                        else:
                                            continue
                                        ref = api.sports_nba_get_games_on_date(cur_time_point[:4], cur_team_name)['result']
                                        if ref:
                                            match_date = list(ref['game_date'].values())
                                            team_home = list(ref['team_name_home'].values())
                                            team_away = list(ref['team_name_away'].values())
                                            pts_home = list(ref['pts_home'].values())
                                            pts_away = list(ref['pts_away'].values())
                                            result_nba = get_sports_db_to_json(match_date, team_home, team_away, pts_home, pts_away)
                                            print('current result', result_nba)
                                            reference.append({'nba games: ': result_nba})
                            except AttributeError:
                                print('multiple teams',planner['team'])
                                all_teams = planner['team']
                                for cur_team in all_teams:
                                    cur_team_name = find_most_similar_item(process_team_name(cur_team), nba_names)

                                    try:
                                        if len(querytime) == 1:
                                            ref = api.sports_nba_get_games_on_date(querytime[0][:4], cur_team_name)['result']
                                            match_date = list(ref['game_date'].values())
                                            team_home = list(ref['team_name_home'].values())
                                            team_away = list(ref['team_name_away'].values())
                                            pts_home = list(ref['pts_home'].values())
                                            pts_away = list(ref['pts_away'].values())
                                            result_nba = get_sports_db_to_json(match_date, team_home, team_away, pts_home, pts_away)
                                            print('current result', result_nba)
                                            reference.append({'nba games: ': result_nba})
                                        elif len(querytime) == 2:
                                            time_points = parse_date_range(querytime)
                                            time_points_set = []
                                            for cur_time_point in time_points:
                                                if not cur_time_point[:4] in time_points_set:
                                                    time_points_set.append(cur_time_point[:4])
                                                else:
                                                    continue
                                                ref = api.sports_nba_get_games_on_date(cur_time_point[:4], cur_team_name)['result']
                                                match_date = list(ref['game_date'].values())
                                                team_home = list(ref['team_name_home'].values())
                                                team_away = list(ref['team_name_away'].values())
                                                pts_home = list(ref['pts_home'].values())
                                                pts_away = list(ref['pts_away'].values())
                                                result_nba = get_sports_db_to_json(match_date, team_home, team_away, pts_home, pts_away)
                                                print('current result', result_nba)
                                                reference.append({'nba games: ': result_nba})
                                    except:
                                        print('error multiple team')
                                        continue

                if planner['sport_type'] == 'soccer':
                    sport_entity_soccer_count += 1
                    date_pattern = r'(\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}[-/]\d{1,2}[-/]?\d{0,2}|\d{4})'
                    #if 'datetime' in planner.keys():
                    #    match_str = re.findall(date_pattern, planner['datetime'])
                    #else:
                    #    match_str = []

                    if 'datetime' in planner.keys():
                        if "~" in planner['datetime']:
                            period_one, period_two = planner['datetime'].split("~")
                            if "past" in period_one:
                                match_str = ["~"+period_two]
                            elif "past" in period_two:
                                match_str = ["~"+period_one]
                            elif "future" in period_one:
                                match_str = [period_two+"~"]
                            elif "future" in period_two:
                                match_str = [period_one+"~"]
                            else:
                                match_str = [period_one+"~~"+period_two]
                        else:
                            found_time = re.findall(date_pattern, planner['datetime'])
                            if found_time:
                                match_str = found_time
                            else:
                                match_str = [query_time]
                    else:
                        match_str = []
                    for cur_date in match_str:
                        if 'team' in planner.keys():
                            if cur_date[0] == "~":
                                date = parse_date(cur_date[1:], query_time)
                                if len(date) == 10:
                                    querytime = [date + " 00:00:00 EST"]
                                else:
                                    querytime = [date]
                            elif cur_date[-1] == "~":
                                date = parse_date(cur_date[:-1], query_time)
                                if len(date) == 10:
                                    querytime = [date + " 00:00:00 EST"]
                                else:
                                    querytime = [date]
                            elif "~~" in cur_date:
                                date1 = parse_date(cur_date.split("~~")[0], query_time)
                                if len(date1) == 10:
                                    querytime1 = date1 + " 00:00:00 EST"
                                else:
                                    querytime1 = date1
                                date2 = parse_date(cur_date.split("~~")[1], query_time)
                                if len(date2) == 10:
                                    querytime2 = date2 + " 00:00:00 EST"
                                else:
                                    querytime2 = date2
                                querytime = [querytime1, querytime2]
                            else:
                                date = parse_date(cur_date, query_time)
                                if len(date) == 10:
                                    querytime = [date + " 00:00:00 EST"]
                                else:
                                    querytime = [date]
                            print(querytime, planner, query)
                            cur_team_name = find_most_similar_item(process_team_name(planner['team']), soccer_names)
                            if len(querytime) == 1:
                                ref = api.sports_soccer_get_games_on_date(querytime[0][:4], cur_team_name)['result']
                                all_dates = []
                                all_matches = []
                                #print(ref, querytime, planner)
                                for item in ref['date']:
                                    cur_date = item.split(',')[-1].split()[0][1:]
                                    cur_match = item.split(',')[-1].split(cur_date)[-1][:-2]
                                    all_dates.append(cur_date)
                                    all_matches.append(cur_match)
                                all_GFs = list(ref['GF'].values())
                                all_GAs = list(ref['GA'].values())
                                host_team_captains = list(ref['Captain'].values())
                                result_soccer = get_sports_soccer_db_to_json(all_dates, all_matches, all_GFs, all_GAs, host_team_captains)
                                reference.append({'soccer games: ': result_soccer})
                            elif len(querytime) == 2:
                                time_points = parse_date_range(querytime)
                                time_points_set = []
                                for cur_time_point in time_points:
                                    if not cur_time_point[:4] in time_points_set:
                                        time_points_set.append(cur_time_point[:4])
                                    else:
                                        continue
                                    ref = api.sports_soccer_get_games_on_date(cur_time_point[:4], cur_team_name)['result']
                                    all_dates = []
                                    all_matches = []
                                    for item in ref['date']:
                                        cur_date = item.split(',')[-1].split()[0][1:]
                                        cur_match = item.split(',')[-1].split(cur_date)[-1][:-2]
                                        all_dates.append(cur_date)
                                        all_matches.append(cur_match)
                                    all_GFs = list(ref['GF'].values())
                                    all_GAs = list(ref['GA'].values())
                                    host_team_captains = list(ref['Captain'].values())
                                    result_soccer = get_sports_soccer_db_to_json(all_dates, all_matches, all_GFs, all_GAs, host_team_captains)
                                    reference.append({'soccer games: ': result_soccer})
            if len(reference) == 0:
                sport_miss_ref_count += 1
                print('sport entity miss')
            print(query)
            print(planner)
        
        if domain == 'music':
            music_count += 1
            if 'artist_name' in planner.keys() and planner['artist_name'] is not None:
                music_entity_artist_count += 1
                artist_names = planner['artist_name'].split(',') if isinstance(planner['artist_name'], str) else planner['artist_name']
                for artist_name in artist_names:
                    flag = False
                    try:
                        top_artist_name = api.music_search_artist_entity_by_name(artist_name)['result'][0]
                        if not top_artist_name.lower() in query:
                            flag = True
                        ref = api.music_get_members(top_artist_name)['result']
                        reference.append({top_artist_name + '_' + 'member': ref})
                        ref = api.music_get_artist_birth_place(top_artist_name)['result']
                        reference.append({top_artist_name + '_' + 'birth place': ref})
                        ref = api.music_get_artist_birth_date(top_artist_name)['result']
                        reference.append({top_artist_name + '_' + 'birth date': ref})
                        ref = api.music_get_lifespan(top_artist_name)['result']
                        reference.append({top_artist_name + '_' + 'lifespan': ref})
                        all_songs = api.music_get_artist_all_works(top_artist_name)['result']
                        ref = []
                        for cur_song in all_songs:
                            try:
                                cur_song_date = api.music_get_song_release_date(cur_song)['result']
                                ref.append(cur_song + ' (released on '+cur_song_date+')')
                            except:
                                ref.append(cur_song)
                                continue
                        reference.append({top_artist_name + '_' + 'artist work': ref})
                        ref = api.music_grammy_get_award_count_by_artist(top_artist_name)['result']
                        reference.append({top_artist_name + '_' + 'grammy award count': ref})
                        ref = api.music_grammy_get_award_date_by_artist(top_artist_name)['result']
                        reference.append({top_artist_name + '_' + 'grammy award date': ref})
                        billboard_list = api.music_get_billboard_rank_date(1)[1]
                        billboard_list_lower = [item.lower() for item in billboard_list]
                        ref = billboard_list.count(artist_name)
                        reference.append({artist_name + '_' + 'billboard top-1 count': ref})
                        
                    except:
                        print('no music original')
                    if flag:
                        print('multi music')
                        try:
                            top_entity_name = api.open_search_entity_by_name(artist_name)['result'][0]
                            ref = api.open_get_entity(top_entity_name)['result']
                            reference.append({top_entity_name: ref})
                        except:
                            print('no open in music')
                        try:
                            ref = api.movie_get_person_info(artist_name)['result'][0]
                            if 'acted_movies' in ref.keys():
                                current_acted = []
                                cur_film_ids = ref['acted_movies']
                                for cur_film_id in cur_film_ids:
                                    try:
                                        movie_info = api.movie_get_movie_info_by_id(int(cur_film_id))['result']
                                        movie_name = movie_info['title']
                                        movie_release = movie_info['release_date']
                                        movie_box = movie_info['revenue']
                                        if movie_box == 0:
                                            movie_box = 'no information'
                                        current_info = movie_name + " (released on " + movie_release + ", box office: " +movie_box+ ")"
                                        current_acted.append(current_info)
                                    except:
                                        print('no id', cur_film_id)
                                        continue
                                ref['acted_movies'] = current_acted
                            if 'directed_movies' in ref.keys():
                                current_directed = []
                                cur_film_ids = ref['directed_movies']
                                for cur_film_id in cur_film_ids:
                                    try:
                                        movie_info = api.movie_get_movie_info_by_id(int(cur_film_id))['result']
                                        movie_name = movie_info['title']
                                        movie_release = movie_info['release_date']
                                        movie_box = movie_info['revenue']
                                        if movie_box == 0:
                                            movie_box = 'no information'
                                        current_info = movie_name + " (released on " + movie_release + ", box office: " +movie_box+ ")"
                                        current_directed.append(current_info)
                                    except:
                                        print('no id', cur_film_id)
                                        continue
                                ref['directed_movies'] = current_directed
                            print('updated ref', ref)
                            reference.append({person + '_' + 'person information': ref})
                        except:
                            print('no movie in music')

            if 'song_name' in planner.keys() and planner['song_name'] is not None:
                music_entity_song_count += 1
                song_names = planner['song_name'].split(',') if isinstance(planner['song_name'], str) else planner['song_name']
                for song_name in song_names:
                    top_song_name = api.music_search_song_entity_by_name(song_name)['result'][0]
                    ref = api.music_get_song_author(top_song_name)['result']
                    reference.append({top_song_name + '_' + 'author': ref})
                    ref = api.music_grammy_get_award_count_by_song(top_song_name)['result']
                    reference.append({top_song_name + '_' + 'grammy award count': ref})
                    ref = api.music_get_song_release_country(top_song_name)['result']
                    reference.append({top_song_name + '_' + 'release country': ref})
                    ref = api.music_get_song_release_date(top_song_name)['result']
                    reference.append({top_song_name + '_' + 'release date': ref})
            
            if len(reference) == 0:
                music_miss_ref_count += 1
                print('music entity miss')
            print(query)
            print(planner)
    
    reference_str = '<DOC>\n'.join([str(i) for i in reference])
    res.append(
         {
                'interaction_id': response['interaction_id'],
                'query': query,
                'query_time': response['query_time'],
                'query_extract': planner,
                'kg_response_str': reference_str,
                'kg_response_trimmed': trim_predictions_to_token_length(reference_str, 2000)
        }
    )


print('total questions and domain hit count', total_count, domain_hit_count)
print('open, finance, movie, sport, music', open_count, finance_count, movie_count, sport_count, music_count)
print('open entity count', open_entity_count)
print('finance entity count', finance_entity_count)
print('movie entity count', movie_entity_movie_count, movie_entity_person_count, movie_entity_year_count)
print('sport entity count', sport_entity_nba_count, sport_entity_soccer_count)
print('music entity count', music_entity_artist_count, music_entity_song_count)
print('open, finance, movie, sport, music miss ref count', open_miss_ref_count, finance_miss_ref_count, movie_miss_ref_count, sport_miss_ref_count, music_miss_ref_count)
with open(kg_res_path, 'w') as f:
    json.dump(res, f)
