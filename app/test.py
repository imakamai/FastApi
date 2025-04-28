import os
import chromadb
import uuid

from nltk import ngrams
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from openai import OpenAI
from rouge_score import rouge_scorer


reader = PdfReader("C:\\Users\\imakamai\\Desktop\\CoverLetter.pdf")
number_of_pages = len(reader.pages)
page = reader.pages[0]
text = page.extract_text()
print(text.split(".\n"))


client = chromadb.PersistentClient(path="D:\\Python\\FastAPI Bridge Project\\chromadb")

model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

data = text.split(".\n")

documents = []
embeddings = []
metadatas = []
ids = []
i = 0

for text in data:
    documents.append(text)
    embeddings.append(model.encode(text))
    metadatas.append({"metadata": text})
    ids.append(str(uuid.uuid4()))
    i =+ 1

collection_name = "pet_collection_emb"
try:
    pet_collection_emb = client.get_collection(collection_name)
except:
    pet_collection_emb = client.create_collection(collection_name)
# pet_collection_emb = client.get_collection("pet_collection_emb")
#
# pet_collection_emb.add(
#     documents=documents,
#     embeddings=embeddings,
#     metadatas=metadatas,
#     ids=ids
# )


query = "Which univerzity she finished?"

input_em = model.encode(query).tolist()


results = pet_collection_emb.query(
    query_embeddings=[input_em],
    n_results=2
)

retrieved_docs = "\n".join(results['documents'][0])

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that answers questions based on provided context."),
        ("human", f"Context:\n{retrieved_docs}\n\nQuestion: {query}")
    ]
)

formatted_prompt = prompt.format_messages()

chat = ChatOpenAI(model_name="gtp-3.5-turbo")

response = chat(formatted_prompt)
print("\n Answer:")
print(response.content)

