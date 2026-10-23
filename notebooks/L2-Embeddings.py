# %% [markdown]
# Source notebook: L2-Embeddings.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # Lesson 2: Embeddings

# %% [markdown]
# Note: The numeric values of embeddings you see in your notebook may vary slightly from those filmed.

# %% [markdown]
# ### Setup
# Load needed API keys and relevant Python libaries.

# %%
# Notebook cell 4
# !pip install cohere umap-learn altair datasets

# %%
# Notebook cell 5
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# %%
# Notebook cell 6
import cohere
co = cohere.Client(os.environ['COHERE_API_KEY'])

# %%
# Notebook cell 7
import pandas as pd

# %% [markdown]
# ## Word Embeddings
# 
# Consider a very small dataset of three words.

# %%
# Notebook cell 9
three_words = pd.DataFrame({'text':
  [
      'joy',
      'happiness',
      'potato'
  ]})

three_words

# %% [markdown]
# Let's create the embeddings for the three words:
# You may see an 'unknown field' warning which can be ignored.

# %%
# Notebook cell 11
three_words_emb = co.embed(texts=list(three_words['text']),
                           model='embed-english-v2.0').embeddings

# %%
# Notebook cell 12
word_1 = three_words_emb[0]
word_2 = three_words_emb[1]
word_3 = three_words_emb[2]

# %%
# Notebook cell 13
word_1[:10]

# %% [markdown]
# ## Sentence Embeddings

# %% [markdown]
# Consider a very small dataset of three sentences.

# %%
# Notebook cell 16
sentences = pd.DataFrame({'text':
  [
   'Where is the world cup?',
   'The world cup is in Qatar',
   'What color is the sky?',
   'The sky is blue',
   'Where does the bear live?',
   'The bear lives in the the woods',
   'What is an apple?',
   'An apple is a fruit',
  ]})

sentences

# %% [markdown]
# Let's create the embeddings for the three sentences:

# %%
# Notebook cell 18
emb = co.embed(texts=list(sentences['text']),
               model='embed-english-v2.0').embeddings

# Explore the 10 first entries of the embeddings of the 3 sentences:
for e in emb:
    print(e[:3])

# %%
# Notebook cell 19
len(emb[0])

# %%
# Notebook cell 20
#import umap
#import altair as alt

# %% [markdown]
# The next code cell is for hiding some warnings that appear when importing the `umap_plot` library.

# %%
# Notebook cell 22
# hide the warnings that would appear when importing the UMAP library
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# %%
# Notebook cell 23
from utils import umap_plot

# %%
# Notebook cell 24
chart = umap_plot(sentences, emb)

# %%
# Notebook cell 25
chart.interactive()

# %% [markdown]
# ## Articles Embeddings

# %%
# Notebook cell 27
import pandas as pd
wiki_articles = pd.read_pickle('wikipedia.pkl')
wiki_articles

# %%
# Notebook cell 28
import numpy as np
from utils import umap_plot_big

# %%
# Notebook cell 29
articles = wiki_articles[['title', 'text']]
embeds = np.array([d for d in wiki_articles['emb']])

chart = umap_plot_big(articles, embeds)
chart.interactive()

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

