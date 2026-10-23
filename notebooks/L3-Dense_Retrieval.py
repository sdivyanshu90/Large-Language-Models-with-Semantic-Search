# %% [markdown]
# Source notebook: L3-Dense_Retrieval.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

from IPython import get_ipython as _active_ipython

def get_ipython():
    """Use the active shell and expose this script namespace to line magics."""
    shell = _active_ipython()
    if shell is None:
        raise RuntimeError("This source uses IPython commands; run it with ipython instead of python.")
    import inspect
    caller = inspect.currentframe().f_back
    shell.user_ns.update(caller.f_globals)
    shell.user_ns.update(caller.f_locals)
    return shell

# %%
# Notebook cell 1
get_ipython().system('pip install cohere')
get_ipython().system('pip install weaviate-client Annoy')
# !pip install newspaper3k

# %%
# Notebook cell 2
def print_result(result):
    """ Print results with colorful formatting """
    for i,item in enumerate(result):
        print(f'item {i}')
        for key in item.keys():
            print(f"{key}:{item.get(key)}")
            print()
        print()


### Revised version
def keyword_search(query, 
                   client,
                   results_lang='en', 
                   properties = ["title","url","text"],
                   num_results=3):

    where_filter = {
    "path": ["lang"],
    "operator": "Equal",
    "valueString": results_lang
    }

    response = (
        client.query.get("Articles", properties)
        .with_bm25(
          query=query
        )
        .with_where(where_filter)
        .with_limit(num_results)
        .do()
        )

    result = response['data']['Get']['Articles']
    return result

# %%
# Notebook cell 3
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
cohere_api_key = user_secrets.get_secret("COHERE_API_KEY")
gemini_api_key = user_secrets.get_secret("GOOGLE_API_KEY")
weaviate_api_key = user_secrets.get_secret("WEAVIATE_API_KEY")
weaviate_api_url = user_secrets.get_secret("WEAVIATE_API_URL")

# %%
# Notebook cell 4
import cohere
co = cohere.Client(cohere_api_key)

# %%
# Notebook cell 5
import weaviate
print(weaviate.__version__)

# %%
# Notebook cell 6
import weaviate
auth_config = weaviate.auth.AuthApiKey(
    api_key=weaviate_api_key)

# %%
# Notebook cell 7
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_api_url,
    auth_credentials=auth_config,
    headers={
        "X-Cohere-Api-Key": cohere_api_key,
    }
)

# %%
# Notebook cell 8
client.is_ready() #check if True

# %% [markdown]
# ## Part 1: Vector Database for semantic Search

# %%
# Notebook cell 10
from weaviate.classes.query import Filter, MetadataQuery

def dense_retrieval(query, 
                    results_lang='en', 
                    properties=["text", "title", "url", "views", "lang"],
                    num_results=5):

    # Get the collection (replace 'Articles' with your actual class name if different)
    collection = client.collections.get("Articles")

    # Build the filter to restrict by language
    where_filter = Filter.by_property("lang").equal(results_lang)

    # Perform the semantic search with near_text
    response = collection.query.near_text(
        query=query,
        filters=where_filter,
        return_properties=properties,
        return_metadata=MetadataQuery(distance=True),
        limit=num_results
    )

    # Collect and return the result objects
    results = [
        {
            **obj.properties,
            "distance": obj.metadata.distance  # if needed
        }
        for obj in response.objects
    ]

    return results

# %%
# Notebook cell 11
print(client.collections.list_all())

# %% [markdown]
# ### Basic Query

# %%
# Notebook cell 13
# query = "Who wrote Hamlet?"
# dense_retrieval_results = dense_retrieval(query)
# print_result(dense_retrieval_results)

# %% [markdown]
# ### Medium Query

# %%
# Notebook cell 15
# query = "What is the capital of Canada?"
# dense_retrieval_results = dense_retrieval(query)
# print_result(dense_retrieval_results)

# %%
# Notebook cell 16
# query = "What is the capital of Canada?"
# keyword_search_results = keyword_search(query, client)
# print_result(keyword_search_results)

# %% [markdown]
# ### Complicated Query

# %%
# Notebook cell 18
# query = "Tallest person in history?"
# keyword_search_results = keyword_search(query, client)
# print_result(keyword_search_results)

# %%
# Notebook cell 19
# query = "Tallest person in history"
# dense_retrieval_results = dense_retrieval(query)
# print_result(dense_retrieval_results)

# %%
# Notebook cell 20
# query = "أطول رجل في التاريخ"
# dense_retrieval_results = dense_retrieval(query)
# print_result(dense_retrieval_results)

# %%
# Notebook cell 21
# query = "film about a time travel paradox"
# dense_retrieval_results = dense_retrieval(query)
# print_result(dense_retrieval_results)

# %% [markdown]
# ## Part 2: Building Semantic Search from Scratch

# %% [markdown]
# ### Get the text archive:

# %%
# Notebook cell 24
from annoy import AnnoyIndex
import numpy as np
import pandas as pd
import re

# %%
# Notebook cell 25
text = """
Interstellar is a 2014 epic science fiction film co-written, directed, and produced by Christopher Nolan.
It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, Matt Damon, and Michael Caine.
Set in a dystopian future where humanity is struggling to survive, the film follows a group of astronauts who travel through a wormhole near Saturn in search of a new home for mankind.

Brothers Christopher and Jonathan Nolan wrote the screenplay, which had its origins in a script Jonathan developed in 2007.
Caltech theoretical physicist and 2017 Nobel laureate in Physics[4] Kip Thorne was an executive producer, acted as a scientific consultant, and wrote a tie-in book, The Science of Interstellar.
Cinematographer Hoyte van Hoytema shot it on 35 mm movie film in the Panavision anamorphic format and IMAX 70 mm.
Principal photography began in late 2013 and took place in Alberta, Iceland, and Los Angeles.
Interstellar uses extensive practical and miniature effects and the company Double Negative created additional digital effects.

Interstellar premiered on October 26, 2014, in Los Angeles.
In the United States, it was first released on film stock, expanding to venues using digital projectors.
The film had a worldwide gross over $677 million (and $773 million with subsequent re-releases), making it the tenth-highest grossing film of 2014.
It received acclaim for its performances, direction, screenplay, musical score, visual effects, ambition, themes, and emotional weight.
It has also received praise from many astronomers for its scientific accuracy and portrayal of theoretical astrophysics. Since its premiere, Interstellar gained a cult following,[5] and now is regarded by many sci-fi experts as one of the best science-fiction films of all time.
Interstellar was nominated for five awards at the 87th Academy Awards, winning Best Visual Effects, and received numerous other accolades"""

# %% [markdown]
# ### Chunking: 

# %%
# Notebook cell 27
# Split into a list of sentences
texts = text.split('.')

# Clean up to remove empty spaces and new lines
texts = np.array([t.strip(' \n') for t in texts])

# %%
# Notebook cell 28
texts

# %%
# Notebook cell 29
# Split into a list of paragraphs
texts = text.split('\n\n')

# Clean up to remove empty spaces and new lines
texts = np.array([t.strip(' \n') for t in texts])

# %%
# Notebook cell 30
texts

# %%
# Notebook cell 31
# Split into a list of sentences
texts = text.split('.')

# Clean up to remove empty spaces and new lines
texts = np.array([t.strip(' \n') for t in texts])

# %%
# Notebook cell 32
title = 'Interstellar (film)'

texts = np.array([f"{title} {t}" for t in texts])

# %%
# Notebook cell 33
texts

# %% [markdown]
# ### Get the embeddings:

# %%
# Notebook cell 35
response = co.embed(
    texts=texts.tolist()
).embeddings

# %%
# Notebook cell 36
embeds = np.array(response)
embeds.shape

# %% [markdown]
# ### Create the search index:

# %%
# Notebook cell 38
search_index = AnnoyIndex(embeds.shape[1], 'angular')
# Add all the vectors to the search index
for i in range(len(embeds)):
    search_index.add_item(i, embeds[i])

search_index.build(10) # 10 trees
search_index.save('test.ann')

# %%
# Notebook cell 39
pd.set_option('display.max_colwidth', None)

def search(query):

  # Get the query's embedding
  query_embed = co.embed(texts=[query]).embeddings

  # Retrieve the nearest neighbors
  similar_item_ids = search_index.get_nns_by_vector(query_embed[0],
                                                    3,
                                                  include_distances=True)
  # Format the results
  results = pd.DataFrame(data={'texts': texts[similar_item_ids[0]],
                              'distance': similar_item_ids[1]})

  print(texts[similar_item_ids[0]])

  return results

# %%
# Notebook cell 40
query = "How much did the film make?"
search(query)

# %%
# Notebook cell 41

