#!pip install lark pinecone-client tiktoken

import keys

import os
import sys
import datetime

import pinecone

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Save chat history
def save_data(chat_history):
    with open(chatlog + 'chat_history_self_query_v2.txt', 'a') as f:
        f.write('#' * 80)
        f.write('\n' + str(datetime.datetime.now()) + '\n')
        for item in chat_history:
            # print(item)
            f.write("Q: %s\n" % item[0])
            f.write("A: %s\n\n" % item[1])
        f.write('#' * 80 + '\n')


##############################################################################
# Setup
##############################################################################
knowledge_base = 'knowledge_base/' # directory of pdfs
chatlog = 'chatlog/' # directory of chatlogs

os.environ["OPENAI_API_KEY"] = keys.openai_api
embeddings = OpenAIEmbeddings()

pinecone.init(api_key=keys.pinecone_api, environment=keys.pinecone_env)

setup = input("Do you want to setup a new index? (y/n) ")

while setup not in ['y', 'n']:
    setup = input("Do you want to setup a new index? (y/n) ")

if setup == 'y':
    print("Setting up new index...")

    #create index
    pinecone.create_index(name=keys.pinecone_index, dimension=1536)

    #get data
    loader = DirectoryLoader(knowledge_base, glob='*.pdf', show_progress=True, loader_cls=PyPDFLoader)
    documents = loader.load_and_split()

    print(type(documents[0]))
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000)
    # docs = text_splitter.split_text(documents)


    #insert data
    vectorstore = Pinecone.from_documents(documents=documents, embedding=embeddings, index_name="test1")

elif setup == 'n':
    #load index
    print("Loading index...")
    vectorstore = Pinecone.from_existing_index(index_name=keys.pinecone_index, embedding=embeddings)

# Chat setup
llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["query"],
    template = "Q: Can you briefly suggest a solution to this problem: \"{query}\" as a list of bullet points? \nA: ")
chain = LLMChain(llm=llm, prompt=prompt)

#query data
history = []


while True:
    query = None
    if not query:
        query = input(">> ")
    if query == 'exit':
        save_data(history)
        sys.exit()
    
    answer = chain.run(query)
    print(answer, "\n\n")

    result = vectorstore.similarity_search_with_relevance_scores(query, k=3)

    for i in result:
        if i[1] < 0.2:
            print(i[0].metadata, '\nSCORE:\t', i[1], '\n\n')
            
        else:
            print('No results found / low confidence\n\n')

    print("#" * 80, '\n\n')
    history.append((query, answer, result))

    
    # result = vectorstore.similarity_search_with_relevance_scores(query, k=1)
    # results = []

    # for i in result:
    #     if i[1] < 0.2:
    #         print(i[0].metadata, '\nSCORE:\t', i[1], '\n\n')
    #         answer = chain.run(i[0].page_content)
    #         print(answer, "\n\n")
    #         results.append((i[0].metadata, answer))
    #     else:
    #         print('No results found / low confidence\n\n')

    # print("#" * 80, '\n\n')
    # history.append((query, results))