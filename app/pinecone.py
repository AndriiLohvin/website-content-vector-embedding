from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import tiktoken
import openai
from dotenv import load_dotenv
import os
import pinecone

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
tokenizer = tiktoken.get_encoding('cl100k_base')

# pinecone.create_index(os.getenv("PINECONE_ENVIRONMENT"), dimension=1000)

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

embeddings = OpenAIEmbeddings()
similarity_min_value = 0.5


def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def train_text():
    print("train-begin")
    with open("./data/data1.txt", "r") as file:
        content = file.read()
    doc = Document(page_content=content, metadata={"source": "data1.txt"})
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents([doc])
    Pinecone.from_documents(
        chunks, embeddings, index_name=os.getenv("PINECONE_INDEX"))
    print("train-end")


def generate_response_streaming(msg: str):
    print("message" + msg)
    db = Pinecone.from_existing_index(
        index_name=os.getenv("PINECONE_INDEX"), embedding=embeddings)
    results = db.similarity_search(msg, k=3)
    print("results-size: " + str(len(results)))
    context = ""
    for result in results:
        context += f"\n\n{result.page_content}"

    instructor = f"""
        Answer questions based on the sample inputs and outputs, and given context, and as well as your knowledge.
        You must respond within 10-20 words.
        The shorter, the better.
        If context don't include answer that matches question, kindly reply it doesn't have answer.
        -----------------------
        This is context you can refer to.
        {context}
        -----------------------
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        max_tokens=1000,
        messages=[
            {'role': 'system', 'content': instructor},
            {'role': 'user', 'content': msg}
        ],
        # stream=True
    )
    print(response)
    print(response.choices[0].message.content)
