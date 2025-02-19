ZERO_SHOT_PLANNER_TEMPLATE = """
You are an agent only outputs JSON. You are given a Query and Query Time. Do the following: 

1) Determine the domain the query is about. The domain should be one of the following: "finance", "sports", "music", "movie", "encyclopedia". If none of the domain applies, use "other". Use "domain" as the key in the result json. 

2) Extract structured information from the query. Include different keys into the result json depending on the domains, amd put them DIRECTLY in the result json. Here are the rules:

For `encyclopedia` and `other` queries, these are possible keys:
-  `main_entity`: extract the main entity of the query. 

For `finance` queries, these are possible keys:
- `market_identifier`: stock identifiers including individual company names, stock symbols.
- `metric`: financial metrics that the query is asking about. This must be one of the following: `price`, `dividend`, `P/E ratio`, `EPS`, `marketCap`, and `other`.

For `movie` queries, these are possible keys:
- `movie_name`: name of the movie
- `movie_aspect`: if the query is about a movie, which movie aspect the query asks. This must be one of the following: `budget`, `genres`, `original_language`, `original_title`, `release_date`, `revenue`, `title`, `cast`, `crew`, `rating`, `length`.
- `person`: person name related to moves
- `person_aspect`: if the query is about a person, which person aspect the query asks. This must be one of the following: `acted_movies`, `directed_movies`, `oscar_awards`, `birthday`.
- `year`: if the query is about movies released in a specific year, extract the year

For `music` queries, these are possible keys:
- `artist_name`: name of the artist
- `artist_aspect`: if the query is about an artist, extract the aspect of the artist. This must be one of the following: `member`, `birth place`, `birth date`, `lifespan`, `artist work`, `grammy award count`, `grammy award date`.
- `song_name`: name of the song
- `song_aspect`: if the query is about a song, extract the aspect of the song. This must be one of the following: `auther`, `grammy award count`, `release country`, `release date`.

For `sports` queries, these are possible keys:
- `sport_type`: one of `basketball`, `soccer`, `other`
- `tournament`: such as NBA, World Cup, Olympic.
- `team`: teams that user interested in.

Return the results in a FLAT json. 

*NEVER include ANY EXPLANATION or NOTE in the output, ONLY OUTPUT JSON!!!*  

Here are some examples:
### Query
what was the volume of trades for rcm on the last day?
Your extracted JSON should be: {"domain": "finance", "market_identifier": "rcm", "metric": "volume of trades"}

### Query
on 2022-10-11, how many points did bulls put up in their game?
Your extracted JSON should be: {"domain": "sports", "sport_type": "basketball", "team": "chicago bulls"}
"""

COT_TEMPLATE_FOR_DATE = """
Please extract the time frame that user interested in. When datetime is not explicitly mentioned, use `Query Time` as default. Enclose your datetime extracted with <>! Use '~' to indicate time period. 

Here are some examples:
### Query
what was the volume of trades for rcm on the last day?
### Query Time
02/28/2024, 08:04:08 PT
Question: What is the user interested time frame of the Query? Please think step by step.
Your output: Since the query is asking about the last day of the Query Time, the time frame should be <02/27/2024>.

### Query
which team did boston celtics take on in their matchup on 2023-05-29?
### Query Time
03/15/2024, 16:05:17 PT
Question: What is the user interested time frame of the Query? Please think step by step.
Your output: The time frame should be <05/29/2023>, which is explicitly stated in the Query.

### Query
on which date did sgml distribute dividends the first time?
### Query Time
02/28/2024, 08:25:10 PT
Question: What is the user interested time frame of the Query? Please think step by step.
Your output: The time frame should be <02/28/2024>. Since there is no datetime explicitly mentioned and we take the Query Time as the default answer.

### Query
what's the schedule looking like for west ham's next game in eng-premier league?
### Query Time
03/15/2024, 15:48:32 PT
Question: What is the user interested time frame of the Query? Please think step by step.
Your output: The time frame should be <03/15/2024 ~ future>. Since there is no datetime that can be confidently inferred from the query, we take the Query Time to (~) future time period as the answer.

### Query
on average, what was the daily high stock price of xpev over the past week?
### Query Time
02/28/2024, 08:17:40 PT
Question: What is the user interested time frame of the Query? Please think step by step.
Your output: The time frame should be <02/21/2024 ~ 02/28/2024>. Since the question is asking for the value over the past week. We take 02/21/2024 ~ 02/28/2024 time period as the answer. 
"""