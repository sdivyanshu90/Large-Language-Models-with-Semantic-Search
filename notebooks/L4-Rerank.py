# %% [markdown]
# Source notebook: L4-Rerank.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # ReRank

# %% [markdown]
# ## Setup
# 
# Load needed API keys and relevant Python libaries.

# %%
# Notebook cell 3
# !pip install cohere 
# !pip install weaviate-client

# %%
# Notebook cell 4
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# %%
# Notebook cell 5
import cohere
co = cohere.Client(os.environ['COHERE_API_KEY'])

# %%
# Notebook cell 6
import weaviate
auth_config = weaviate.auth.AuthApiKey(
    api_key=os.environ['WEAVIATE_API_KEY'])

# %%
# Notebook cell 7
client = weaviate.Client(
    url=os.environ['WEAVIATE_API_URL'],
    auth_client_secret=auth_config,
    additional_headers={
        "X-Cohere-Api-Key": os.environ['COHERE_API_KEY'],
    }
)

# %% [markdown]
# ## Dense Retrieval

# %%
# Notebook cell 9
from utils import dense_retrieval

# %%
# Notebook cell 10
query = "What is the capital of Canada?"

# %%
# Notebook cell 11
dense_retrieval_results = dense_retrieval(query, client)

# %%
# Notebook cell 12
from utils import print_result

# %%
# Notebook cell 13
print_result(dense_retrieval_results)

# %% [markdown]
# ## Improving Keyword Search with ReRank

# %%
# Notebook cell 15
from utils import keyword_search

# %%
# Notebook cell 16
query_1 = "What is the capital of Canada?"

# %%
# Notebook cell 17
query_1 = "What is the capital of Canada?"
results = keyword_search(query_1,
                         client,
                         properties=["text", "title", "url", "views", "lang", "_additional {distance}"],
                         num_results=3
                        )

for i, result in enumerate(results):
    print(f"i:{i}")
    print(result.get('title'))
    print(result.get('text'))

# %%
# Notebook cell 18
query_1 = "What is the capital of Canada?"
results = keyword_search(query_1,
                         client,
                         properties=["text", "title", "url", "views", "lang", "_additional {distance}"],
                         num_results=500
                        )

for i, result in enumerate(results):
    print(f"i:{i}")
    print(result.get('title'))
    #print(result.get('text'))

# %%
# Notebook cell 19
def rerank_responses(query, responses, num_responses=10):
    reranked_responses = co.rerank(
        model = 'rerank-english-v2.0',
        query = query,
        documents = responses,
        top_n = num_responses,
        )
    return reranked_responses

# %%
# Notebook cell 20
texts = [result.get('text') for result in results]
reranked_text = rerank_responses(query_1, texts)

# %%
# Notebook cell 21
for i, rerank_result in enumerate(reranked_text):
    print(f"i:{i}")
    print(f"{rerank_result}")
    print()

# %% [markdown]
# ## Improving Dense Retrieval with ReRank

# %%
# Notebook cell 23
from utils import dense_retrieval

# %%
# Notebook cell 24
query_2 = "Who is the tallest person in history?"

# %%
# Notebook cell 25
results = dense_retrieval(query_2,client)

# %%
# Notebook cell 26
for i, result in enumerate(results):
    print(f"i:{i}")
    print(result.get('title'))
    print(result.get('text'))
    print()

# %%
# Notebook cell 27
texts = [result.get('text') for result in results]
reranked_text = rerank_responses(query_2, texts)

# %%
# Notebook cell 28
for i, rerank_result in enumerate(reranked_text):
    print(f"i:{i}")
    print(f"{rerank_result}")
    print()

# %%
# Notebook cell 29


# %%
# Notebook cell 30


# %%
# Notebook cell 31


# %%
# Notebook cell 32


# %%
# Notebook cell 33


# %%
# Notebook cell 34


# %%
# Notebook cell 35


# %%
# Notebook cell 36


# %%
# Notebook cell 37


# %%
# Notebook cell 38


# %%
# Notebook cell 39


# %%
# Notebook cell 40


# %%
# Notebook cell 41


# %%
# Notebook cell 42


# %%
# Notebook cell 43


# %%
# Notebook cell 44


# %%
# Notebook cell 45


# %%
# Notebook cell 46


# %%
# Notebook cell 47

