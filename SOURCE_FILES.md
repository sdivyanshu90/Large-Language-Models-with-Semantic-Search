**Notebook source files**

The source companions below preserve notebook cell order and explanations in readable code blocks. Python uses `# %%` cell markers; TypeScript uses `// %%`. The original notebooks retain their saved visualizations and course context. GitHub language statistics exclude `.ipynb` files through `.gitattributes` and count eligible implementation source normally.

Use the original notebook's environment, dependency versions, datasets, credentials and working directory. This conversion does not supply missing course helpers or fill exercise placeholders. It does not execute training, external APIs, shell commands or setup cells during validation. Review setup and execution cells before running a whole lesson.

Run ordinary Python companions with `python "path/to/lesson.py"`, or open their cells in an editor supporting Python percent scripts. Files marked **IPython** retain notebook magics and shell commands: install IPython in the lesson environment and run `ipython "path/to/lesson.py"`. Files marked **async** have an `async def main()` entry point; run the whole script to preserve its event loop. Script validation checks Python syntax, not availability or behavior of the original dependencies.

TypeScript companions preserve the Deno runtime identified by their notebooks. Run them in the original Deno environment with the permissions and dependencies required by that lesson. Repeated notebook declarations use mutable bindings and reassignment in the script. React component variants are separate `.tsx` modules in the adjacent `.components` directory; the lesson's frontend-injection helpers still require the original course environment. No Python translation is implied for TypeScript code.

| Notebook | Source companion | Execution notes |
|---|---|---|
| [notebooks/L1-Keyword_Search.ipynb](notebooks/L1-Keyword_Search.ipynb) | [notebooks/L1-Keyword_Search.py](notebooks/L1-Keyword_Search.py) | Python |
| [notebooks/L2-Embeddings.ipynb](notebooks/L2-Embeddings.ipynb) | [notebooks/L2-Embeddings.py](notebooks/L2-Embeddings.py) | Python |
| [notebooks/L3-Dense_Retrieval.ipynb](notebooks/L3-Dense_Retrieval.ipynb) | [notebooks/L3-Dense_Retrieval.py](notebooks/L3-Dense_Retrieval.py) | IPython |
| [notebooks/L4-Rerank.ipynb](notebooks/L4-Rerank.ipynb) | [notebooks/L4-Rerank.py](notebooks/L4-Rerank.py) | Python |
| [notebooks/L5-Generative_Search-Updated.ipynb](notebooks/L5-Generative_Search-Updated.ipynb) | [notebooks/L5-Generative_Search-Updated.py](notebooks/L5-Generative_Search-Updated.py) | IPython |
