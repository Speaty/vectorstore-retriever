# Loads all PDFs in the knowledge_base directory and uploads them to Pinecone vector store

import os
import sys
import glob
import datetime

import keys

import pinecone
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"] = keys.openai_api # OpenAI API key

# Load file paths from knowledge_base
def load_file_paths(directory):
    files = glob.glob(directory + '/*.pdf')
    return set(files)

# Load file paths of already uploaded files and discard
def files_to_upload(files):
    with open('uploaded.txt', 'r') as f:
        uploaded = set(f.readlines())
    uploaded = [x.strip() for x in uploaded]
    return files.difference(uploaded)

# Append uploaded files to uploaded.txt
def append_to_file(files, file_name):
    with open(file_name, 'a') as f:
        for file in files:
            print(file)
            f.write(file)
            f.write('\n')

# Load and split PDFs into pages, remove any pages that are too short to be useful
# Note: This function takes a long time to run, also some pages are too short to be useful and can be removed in production
def load_and_split(files):
    print("Loading & splitting files...\nThis may take a while if there are many files...")
    docs = []
    if len(files) > 1:
        for file in files:
            docs.extend(PyPDFLoader(file_path=file).load_and_split())
        # print(docs)
        return docs
    else:
        return PyPDFLoader(file_path=files.pop()).load_and_split()

def chat_setup():
    context = "This is a conversation with an AI assistant. The assistant is helpful, creative, clever, and friendly. The assistants job is to summarise learning materials. The AI assistant should only discuss work appropriate topics."
    llm = ChatOpenAI(temperature=0.9) # Temperature is a measure of randomness. 0.0 is deterministic, 1.0 is very random
    prompt = PromptTemplate(
        input_variables=['query'],
        template = "Q: Briefly suggest a solution to the following problem: {query} as a maximum of 5 bullet points."
    )
    model = LLMChain(llm=llm, prompt=prompt)  
    model.run(context)
    return model


def main():
    knowledge_base = 'knowledge_base/' # Directory of PDFs to be uploaded to Pinecone vector store
    embeddings = OpenAIEmbeddings() # OpenAIEmbeddings uses Ada-v2 model ($0.0001 / 1k tokens) - This is really cheap and effective. alternatively could use open source embeddings from hugging face

    # Load PDFs from knowledge_base directory and split them into pages
    files = load_file_paths(knowledge_base)
    to_upload = files_to_upload(files)
    docs = load_and_split(to_upload)

    if len(docs) > 0: # If there are documents to upload
        # create a new Pinecone index, if it doesn't already exist
        pinecone.init(api_key=keys.pinecone_api, environment=keys.pinecone_env) # This line effectively logs you in to Pinecone
        try:
            print("Creating index...")
            pinecone.create_index(name=keys.pinecone_index, dimension=1536) # Creates index with 1536 dimensions
        except:
            print("Index already exists...")
        print("Uploading files...")
        Pinecone.from_documents(documents=docs, embedding=embeddings, index_name=keys.pinecone_index) # Uploads documents to Pinecone vector store
        append_to_file(to_upload, 'uploaded.txt') # Keeps log of uploaded files
    else:
        print("No files to upload...")
    # print(len(docs))
    print("Loading index...")
    vectorstore = Pinecone.from_existing_index(index_name=keys.pinecone_index, embedding=embeddings) # Loads Pinecone vector store

    # Load chatbot
    print("Loading LLM model...")
    chatbot = chat_setup()

    history = [] # Chatbot history
    
    history.append('#'*50 + str(datetime.datetime.now()) + '#'*50 + '\n')
    # Start chatbot loop
    while True:
        print("#"*50)
        query = None
        if not query:
            query = input(">> ")

        if query in ["quit", "exit", "q"]:
            break
        print('\n')
        answer = chatbot.run(query)
        results = vectorstore.similarity_search_with_relevance_scores(query, k=1) # Search Pinecone vector store for similar documents

        for result in results:
            if result[1] < 0.22:
                # print(result[0].page_content)
                print("Source:\t", result[0].metadata['source'], "\nPage:\t", str(result[0].metadata['page']), "\nSimilarity Score:\t", result[1], "\n")
                answer = chatbot.run(result[0].page_content)
                print(answer)

                history.extend([query, '\nSource:', result[0].metadata['source'], str(result[0].metadata['page']), '\n', answer, '\n'])
            else:
                print("No results found...")
    append_to_file(history, 'history.txt') # Keep log of chatbot history

if __name__ == "__main__":
    main()