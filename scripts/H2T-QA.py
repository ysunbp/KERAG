import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from typing import List, Optional
from llama import Llama, Dialog
import json
import random
from tqdm import tqdm
import csv
import wikipedia
import requests
import pickle
import time
from transformers import (
    AutoTokenizer,
    pipeline,
)

from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer
from test_dpr import DPRRetriever

context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").cuda()
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").cuda()


retriever = DPRRetriever(question_model, question_tokenizer, context_model, context_tokenizer)

img_suffix = ['jpg', 'png', 'svg', 'gif']

IN_CONTEXT_TRUE = [["Bangladesh Nationalist Party is the member of which international organization?", "according to the wikipedia page, bangladesh nationalist party is a member of the centrist democrat international.", "Asia Pacific Democrat Union or Centrist Democrat International"],
              ["What patrol aircraft is used by the South African Air Force?", "according to the wikipedia infobox, the patrol aircraft used by the south african air force is the c-47tp.", "C-47 Skytrain"],
              ["What party was split from Communist Refoundation Party?", "according to the wikipedia infobox, the italian communist party (pci) was split from to form the communist refoundation party (prc) in 1991.", "Italian Communist Party"],
              ["What is the stadium where BSG Chemie Leipzig (1950)'s home matches are held?", "alfred-kunze-sportpark (also known as georg-schwarz-sportpark)", "Alfred-Kunze-Sportpark or Georg-Schwarz-Sportpark"],
              ["What is the ending theme of My Papa Pi?", 'the ending theme of my papa pi is "my papa pi" by piolo pascual, pia wurtzbach, and pepe herrera.', "Pia Wurtzbach"],
              ["What is the legislative body in Albanian Kingdom (1928–1939)?", "according to the wikipedia infobox and summary, the legislative body in the albanian kingdom (1928–1939) is the constitutional assembly.", "Parliament of Albania"],
              ["The predecessor of Cirilo Almario is?", "manuel p. del rosario, d.d.", "Manuel del Rosario"],
              ["What is the mouth of Montreal River (Timiskaming District)?", "according to the wikipedia infobox and summary, the mouth of the montreal river (timiskaming district) is lake timiskaming on the ottawa river.", "Timiskaming, Unorganized, West Part, Ontario"],
              ["What significant design was created by Joseph Berlin?", "mograbi cinema, tel aviv.", "Cinema of Israel"],
              ["What patrol aircraft is used by the VPB-127?", "pv-1", "Lockheed Ventura or PV-1"]
              ]

IN_CONTEXT_FALSE = [["What/who influenced Charles Fourier?", "bob black.", "Nicolas-Edme Rétif"],
                    ["Which automobile team had the fastest driver during the 1960 Indianapolis 500?", "ken-paul", "A.J. Watson"],
                    ["Which company owns TV Land?", "paramount global.", "Paramount Media Networks"],
                    ["What language is spoken in Evening (magazine)?", "english", "Japanese language"],
                    ["What is the record label for Cogumelo Records?", "cogumelo records.", "Relapse Records"],
                    ["Jim Pearson was born in which place?", "chatham, ontario, canada.", "Falkirk"],
                    ["What is the format of The Wedge (Australian TV series)?", "the format of the wedge (australian tv series) is a sketch show.", "Stereophonic sound"],
                    ["Who developed Flappy?", "flappy bird was developed by .gears, which is a game development company founded by dong nguyen.", "DB-SOFT"],
                    ["What is Cinematic soul derived from?", "soul music, psychedelic soul, orchestral music, and film score.", "Disco"],
                    ["Which automobile team had the fastest driver during the 1953 Curtis Trophy?", "cooper-bristol.", "Cooper Car Company"]
                ]

from SPARQLWrapper import SPARQLWrapper, JSON
sparql = SPARQLWrapper("https://dbpedia.org/sparql")
with open("/data/Head2Tail/dbpedia_entity.json", "r", encoding='utf8') as f:
    entity = json.load(f)
with open("/data/Head2Tail/dbpedia_relation.json", "r", encoding='utf8') as f:
    relation = json.load(f)

entity_map = {}
entity_to_link_map = {}
relation_to_link_map = {}
for i in range(len(entity)):
    e = entity[i][0].split("/")[-1][:-1]
    entity_to_link_map[e] = entity[i][0]
    if e.lower() not in entity_map:
        entity_map[e.lower()] = []
    entity_map[e.lower()].append(e)
    entity[i] = e.lower()
entity.sort()
relation_list = []
for x in relation:
    r = x[0].split("/")[-1][:-1]
    relation_list.append(r)
    relation_to_link_map[r] = x[0]

relation = relation_list

cachefn = "/data/cache/cache.jsonl"
cache = {}
cache["search"] = {}
cache["summary"] = {}
cache["page"] = {}
cache["html"] = {}
cache["infobox"] = {}
cache["sparql"] = {}
cache["sparql-one-hop"] = {}

with open(cachefn, "a", encoding='utf8') as f:
    pass

with open(cachefn, "r", encoding='utf8') as f:
    l = f.readline()
    while l:
        l = json.loads(l)
        if l[0] == 'sparql':
            cache[l[0]][l[1]+'-'+l[2]] = l[3]
        else:
            cache[l[0]][l[1]] = l[2]
        l = f.readline()

def fuzz_match_partio(x, y):
    m = {}
    dist = -len(x)

    def f(i, j):
        if (i, j) in m:
            return m[(i, j)]
        if i == len(x):
            m[(i, j)] = 0
            return m[(i, j)]
        if j == len(y):
            m[(i, j)] = i - len(x)
            return m[(i, j)]
        m[(i, j)] = -len(x)
        if x[i] == y[j]:
            m[(i, j)] = max(m[(i, j)], f(i + 1, j + 1) + 1)
        m[(i, j)] = max(m[(i, j)], f(i, j + 1) - 1)
        m[(i, j)] = max(m[(i, j)], f(i + 1, j) - 1)
        return m[(i, j)]

    for i in range(len(y) - 1, -1, -1):
        dist = max(dist, f(0, i))
    return dist / len(x) if len(x) > 0 else 1

def get_relation(s, query):
    query = query.lower().replace(" ", "")
    bestrel = None
    bestscore = -1
    for i in range(len(s)):
        score = fuzz_match_partio(s[i].lower(), query)
        if score > bestscore or (score == bestscore and bestrel is not None and len(s[i]) > len(bestrel)):
            bestrel = s[i]
            bestscore = score
    return bestrel, bestscore

def search_helper(s, query):
    l,r = 0,len(s)-1
    while l<=r:
        m = (l+r)//2
        if s[m] < query:
            l = m+1
        else:
            r = m-1
    if l < len(s) and s[l].startswith(query):
        return s[l]
    return None

def search(s, query, jt = "_"):
    matched = []
    query = query.lower().split()
    for i in range(len(query)):
        cq = jt.join(query[i:])
        l,r = 0,len(cq)-1
        while l <= r:
            m = (l+r)//2
            if search_helper(s, cq[:m]) is not None:
                l = m+1
            else:
                r = m-1
        if l > 0:
            matched.append([search_helper(s, cq[:l-1]), l-1, i])
    # i 代表query中的第i个词, l-1是对应的第i个词的结束点
    return matched

def get_entity_relation(query):
    global entity
    global relation
    e = search(entity, query)
    bestscore = [0, 0]
    best = [None, None]
    for i in range(len(e)):
        q = query.split()
        q = " ".join(q[:e[i][2]]) + " " + " ".join(q[e[i][2]:])[e[i][1]:]
        r = get_relation(relation, q)
        we = fuzz_match_partio(e[i][0].lower(), query.replace(" ", "_").lower())
        if len(r[0])*r[1] + e[i][1]*we > sum(bestscore) and len(r[0])*r[1] > 0 and e[i][1]*we > 0:
            bestscore = [len(r[0])*r[1], e[i][1]*we]
            best = [e[i][0], r[0]]
    
    return entity_map[best[0]][0] if best[0] is not None else None, best[1]

def get_entity_relation_topk(query, topK=3):
    global entity
    global relation
    e = search(entity, query)
    bestscore = [0 for i in range(topK)]
    best = [(None, None) for i in range(topK)]
    for i in range(len(e)):
        q = query.split()
        q = " ".join(q[:e[i][2]]) + " " + " ".join(q[e[i][2]:])[e[i][1]:]
        r = get_relation(relation, q)
        we = fuzz_match_partio(e[i][0].lower(), query.replace(" ", "_").lower())
        current_score = (len(r[0])*r[1], e[i][1]*we)
        threshold_score = min(bestscore)
        if current_score[0]+current_score[1] > threshold_score:
            min_idx = bestscore.index(threshold_score)
            bestscore[min_idx] = current_score[0]+current_score[1]

            best[min_idx] = (e[i][0], r[0])
    sorted_score = sorted(bestscore, reverse=True)
    best_indices = [bestscore.index(x) for x in sorted_score]
    updated_best = [best[index] for index in best_indices]
    best = updated_best
    results = []
    for i in range(topK):
        results.append([entity_map[best[i][0]][0] if best[i][0] is not None else None, best[i][1]])
    return results

def get_htt_questions(htt_path):
    with open(htt_path, 'r', encoding='utf8') as f:
        question_set = json.load(f)
    head_set = question_set['head']
    torso_set = question_set['torso']
    tail_set = question_set['tail']
    
    head_questions = [[item[2], item[3], item[0][2], item[0][0], item[0][1]] for item in head_set] # question, label, dbpedia entity of the label, subject_entity
    torso_questions = [[item[2], item[3], item[0][2], item[0][0], item[0][1]] for item in torso_set]
    tail_questions = [[item[2], item[3], item[0][2], item[0][0], item[0][1]] for item in tail_set]

    return head_questions, torso_questions, tail_questions


def llama_answer(
    dialog, generator,
    temperature = 0,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None,
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
    

    dialogs: List[Dialog] = [
        [ {"role": "system", "content": dialog['system']},
            {"role": "user", "content": dialog['user']}]]  
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    output = []
    for dialog, result in zip(dialogs, results):
        output.append(result['generation']['content'])
    return output


def get_sparql_answer(cur_entity_relation):
    global cache
    global sparql, entity_to_link_map, relation_to_link_map
    cur_entity, cur_relation = cur_entity_relation
    if cur_entity and cur_relation:
        if cur_entity+'-'+cur_relation in cache['sparql']:
            return cache['sparql'][cur_entity+'-'+cur_relation]
            
        else:
            dbpedia_entity = entity_to_link_map[cur_entity]
            dbpedia_relation = relation_to_link_map[cur_relation]
            query_core = "{entity} {relation} ?c".format(entity=dbpedia_entity, relation=dbpedia_relation)
            # 编写 SPARQL 查询
            query = "select distinct ?c where{"+query_core+"}"
            #print(query)
            # 设置查询并执行
            try:
                sparql.setQuery(query)
                sparql.setReturnFormat(JSON)
                results = sparql.query().convert()

                #print(results)
                # 输出结果
                results_list = []
                for result in results["results"]["bindings"]:
                    results_list.append(result['c']['value'])
                with open(cachefn, "a", encoding='utf8') as f:
                    record = ["sparql", cur_entity, cur_relation, results_list]
                    f.write(json.dumps(record) + "\n")
                return results_list
            except:
                return []
    else:
        return []
    
def get_sparql_one_hop(cur_entity):
    global sparql, relation, cache, entity_to_link_map
    if cur_entity in cache['sparql-one-hop']:
        return cache['sparql-one-hop'][cur_entity]
    else:
        try:
            cur_entity = entity_to_link_map[cur_entity][1:-1]
            query = f"""
            SELECT?subject?predicate?object
            WHERE {{
                <{cur_entity}>?predicate?object.
            }}
            """
            
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            triples = []
            for item in results['results']['bindings']:
                cur_predicate = item['predicate']['value'].split('/')[-1]
                if not cur_predicate in relation:
                    continue
                #print(item)
                triples.append((cur_entity.split('/')[-1], cur_predicate, item['object']['value'].split('/')[-1]))
            if triples:
                with open(cachefn, "a", encoding='utf8') as f:
                    record = ["sparql-one-hop", cur_entity.split('/')[-1], triples]
                    f.write(json.dumps(record) + "\n")
            return triples
        except:
            return []

###################################################

from collections import Counter

def precompute_ngrams(entity_list, n=3):
    """
    预先计算entity_list中每个实体的N-gram特征
    """
    ngram_cache = {}
    for entity in tqdm(entity_list):
        ngrams = [entity[i:i+n] for i in range(len(entity)-n+1)]
        ngram_cache[entity] = set(ngrams)
    return ngram_cache

def ngram_similarity(str1, str2, ngram_cache, n=3):
    """
    使用预计算的N-gram特征计算相似度
    """
    ngrams1 = set([str1[i:i+n] for i in range(len(str1)-n+1)])
    ngrams2 = ngram_cache[str2]
    
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1) + len(ngrams2) - intersection
    
    similarity = intersection / union if union != 0 else 0
    
    return similarity

def fuzzy_match(query_entity, entity_list, ngram_cache, k=3, n=3):
    """
    使用预计算的N-gram特征进行模糊匹配
    """
    
    # 计算每个实体与查询的相似度
    similarities = [(e, ngram_similarity(query_entity, e, ngram_cache, n)) for e in entity_list]
    
    # 根据相似度排序并返回前k个结果
    similarities.sort(key=lambda x: x[1], reverse=True)
    print(similarities[0])
    return [e[0] for e in similarities[:k]]

def chain_entity_with_slash(entity):
    splitted_candidate = entity.lower().split()
    jt = '_'
    if splitted_candidate:
        cq = ''
        for idx, item in enumerate(splitted_candidate):
            cq += item
            if not idx == len(splitted_candidate)-1:
                cq += jt
    return cq

def compose_llm_filter_template(cur_question, candidates):
    template = {}
    template['system'] = "You are a professional semantic parsing expert. Please provide your answer by giving brief (entity, predicate) pair. If no appropriate (entity, predicate) pair can be extracted, please respond (No, No)"
    template['user'] = "[Instruction]: The task is to determine the most appropriate (entity, predicate) pair based on a given question and a list of potential (entity, predicate) pair list extracted by a fuzzy-matching-based semantic parser. MUST NOT provide answer to the question directly. "+'\n'+"[EXAMPLE]: Question: What aircraft bomber was used by the South Vietnam Air Force? Pair List: [('Vietnam', 'Air_Force'), ('South_Vietnam_Air_Force', 'aircraftBomber')]; Your Answer: ('South_Vietnam_Air_Force', 'aircraftBomber')"+'\n'+"[TASK]: Select the most appropriate (entity, predicate) based on this question and entity pair list: Question: "+cur_question+" Pair List: "+str(candidates)
    return template

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

def kb_one_hop_qa(question, naive_triples, generator):
    dialog = {}
    dialog['system'] = 'Please provide a brief answer as short as possible to the question based on your own knowledge and the following relevant TRIPLEs (subject, predicate, object) from DBpedia. Answer "I don\'t know" if you are not confident of your answer.'
    dialog['user'] = ""
    for cur_triple in naive_triples:
        dialog['user'] += 'TRIPLE: '
        dialog['user'] += str(cur_triple)
        dialog['user'] += '\n'
    dialog['user'] += question
    llama_current_answer = llama_answer(dialog, generator, temperature = 0)[0].strip().lower()
    print(dialog['user'])
    return llama_current_answer

def get_list_triple_index(triples, answer, relation_link):
    relation = relation_link.split('/')[-1][:-1]
    for index, item in enumerate(triples):
        if item[2].replace("_", " ").lower() == answer.lower() and item[1] == relation:
            return index
    return -1

def convert_camel_case_to_words(camel_case_str):
    words = []
    current_word = ""
    for char in camel_case_str:
        if char.isupper():
            if current_word:
                words.append(current_word)
            current_word = char.lower()
        else:
            current_word += char
    if current_word:
        words.append(current_word)
    return " ".join(words)

def KERAG(question_set, outpath, generator, ngram_cache, topK=10, dpr_one_hop = 30):
    global entity
    random.seed(20)
    question_set = random.sample(question_set, 375)
    miss = 0
    correct = 0
    auto_eval_correct = 0
    auto_eval_incorrect = 0
    total = 0

    with open(outpath, 'a', newline='') as outfile:
        for cur_question, answer, dbpedia_answer, subject_entity, relation_link in tqdm(question_set):
            total += 1
            entity_relations = get_entity_relation_topk(cur_question, topK=topK) # 这个地方改成topk
            cur_template = compose_llm_filter_template(cur_question, entity_relations)
            cur_entity_relation = entity_relations[:1][0]
            cur_naive_matching = get_sparql_answer(cur_entity_relation)
            if cur_naive_matching:
                cur_detected_entity = cur_entity_relation[0]
                print('cur detected from fuzzy', cur_detected_entity, 'gt', subject_entity)
                naive_triples = get_sparql_one_hop(cur_detected_entity)
            else: # if sparql answer is empty, we use llama to get the subject entity
                llama_response = list(llama_answer(cur_template, generator, temperature = 0)[0].strip().split(', ')) # debug
                llama_entity = llama_response[0][2:-1]
                llama_entity = llama_entity.replace("\\'s", "'s")
                cur_detected_entity = llama_entity
                if '__' in cur_detected_entity:
                    cur_detected_entity = cur_detected_entity.split('__')[0]
                if cur_detected_entity.lower() in entity:
                    cur_detected_entity = cur_detected_entity
                else:
                    splitted_candidate = cur_detected_entity.lower().split()
                    jt = '_'
                    if splitted_candidate:
                        cq = ''
                        for idx, item in enumerate(splitted_candidate):
                            cq += item
                            if not idx == len(splitted_candidate)-1:
                                cq += jt
                        if cq in entity:
                            cur_detected_entity = cq
                        else:
                            cur_detected_entity = fuzzy_match(cur_detected_entity, entity, ngram_cache, 1, 3)[0]
                    else:
                        cur_detected_entity = fuzzy_match(cur_detected_entity, entity, ngram_cache, 1, 3)[0]
                print('cur detected from llama:', cur_detected_entity, 'gt:', subject_entity)
                naive_triples = get_sparql_one_hop(cur_detected_entity)
            naive_triples = list(set(tuple(row) for row in naive_triples if not row[2].split(".")[-1] in img_suffix)) # filter out the images
            print('number of triples obtained', len(naive_triples))
            
            cur_detected_entity = cur_detected_entity.replace('_', " ")
            print('cur_cleaned_entity', cur_detected_entity)
            if len(naive_triples) > dpr_one_hop:
                naive_triples_string = [convert_camel_case_to_words(item[1]) for item in naive_triples]
                cur_replaced_question = cur_question.lower().replace(cur_detected_entity.lower(), '').replace(cur_detected_entity.split('(')[0].lower(), '')
                top_k_indices_and_scores = retriever.retrieve_top_k(cur_replaced_question, naive_triples_string, k=dpr_one_hop)
                top_k_results = [naive_triples[item] for item, score in top_k_indices_and_scores]
                naive_triples = top_k_results
                print(naive_triples)
            elif len(naive_triples) > 0:
                naive_triples_string = [convert_camel_case_to_words(item[1]) for item in naive_triples]
                cur_replaced_question = cur_question.lower().replace(cur_detected_entity.lower(), '').replace(cur_detected_entity.split('(')[0].lower(), '')
                top_k_indices_and_scores = retriever.retrieve_top_k(cur_replaced_question, naive_triples_string, k=len(naive_triples))
                top_k_results = [naive_triples[item] for item, score in top_k_indices_and_scores]
                naive_triples = top_k_results
                print(naive_triples)
            
            
            
            try:
                llm_answer = kb_one_hop_qa(cur_question, naive_triples, generator)
            except:
                flag = True
                tic = time.time()
                while flag:
                    try:
                        naive_triples = naive_triples[:int(len(naive_triples)/2)]
                        llm_answer = kb_one_hop_qa(cur_question, naive_triples, generator)
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
            item["question"] = cur_question
            item["answer"] = answer
            item["llm_answer"] = llm_answer
            json_line = json.dumps(item)
            outfile.write(json_line + '\n')
            if (answer.lower() in llm_answer) or (answer.split("(")[0].strip().lower() in llm_answer) or (answer.lower() in llm_answer.replace('_', " ")):
                correct += 1
                auto_eval_correct += 1
            else:
                if 'don\'t know' in llm_answer:
                    miss += 1
                else:
                    eval_template = compose_eval_template(cur_question, answer.lower(), llm_answer)
                    llama_current_answer = llama_answer(eval_template, evaluator, temperature = 0)[0].strip().lower()
                    if 'no' in llama_current_answer:
                        print('llama no eval', llama_current_answer)
                        auto_eval_incorrect += 1
                    else:
                        auto_eval_correct += 1
    print('+++++++++++++++++++++++++++++++')
    print('Now checking dpr hop', dpr_one_hop)
    print('total', total, 'correct', correct, 'miss', miss, 'accuracy', correct/total, 'miss rate', miss/total)
    print('auto eval accuracy', auto_eval_correct/total)
    print('hallucination rate', 1-miss/total-auto_eval_correct/total)
    print("trustfulness score", auto_eval_correct/total-(1-miss/total-auto_eval_correct/total))
    print('+++++++++++++++++++++++++++++++')
    

if __name__ == "__main__":
    htt_path = "/data/Head2Tail/head_to_tail_dbpedia.json"
    head_questions, torso_questions, tail_questions = get_htt_questions(htt_path)
    cur_model_dir = '/llama3/Meta-Llama-3-8B-Instruct/'
    generator = Llama.build(
            ckpt_dir=cur_model_dir,
            tokenizer_path=cur_model_dir+'tokenizer.model',
            max_seq_len=8192,
            max_batch_size=6)
    
    ngram_cache_path = '/data/cache/3gram_db_entity.pkl'
    with open(ngram_cache_path, 'rb') as f:
        ngram_cache = pickle.load(f)
    
    dpr_hops = [30]
    for dpr_hop in dpr_hops:
        print('Checking head questions')
        KERAG(head_questions, "/result/Head2Tail/xxx-head-8b.json", generator, ngram_cache, topK=10, dpr_one_hop=dpr_hop)
        print('Checking torso questions')
        KERAG(torso_questions,  "/result/Head2Tail/xxx-torso-8b.json", generator, ngram_cache, topK=10, dpr_one_hop=dpr_hop)
        print('Checking tail questions')
        KERAG(tail_questions,  "/result/Head2Tail/xxx-tail-8b.json", generator, ngram_cache, topK=10, dpr_one_hop=dpr_hop)
    
# python -m torch.distributed.run --nproc_per_node 1 H2T-QA.py
