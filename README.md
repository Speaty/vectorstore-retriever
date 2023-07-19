# vectorstore-retriever
CLI application to search a vectorstore for relevant PDF files for a given prompt.

## Setup
1. Enter Pinecone and OpenAI API keys into `keys.py.config` and change the file name to `keys.py`
2. run the command `pip install lark pinecone-client tiktoken` 
3. If training, point to a directory of PDFs in `self_query_retreiver.py`
4. If saving chatlogs, point to a directory in `self_query_retreiver.py`
