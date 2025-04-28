import chromadb
import uuid

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate



import os
import uuid
from PyPDF2 import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import torch
from rouge import RougeMetricsEnglish

# Load PDF file
reader = PdfReader("C:\\Users\\imakamai\\Desktop\\CoverLetter.pdf")
number_of_pages = len(reader.pages)
page = reader.pages[0]
text = page.extract_text()
# Split text into chunks
data = text.split(".\n")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="D:\\Python\\FastAPI Bridge Project\\chromadb")

# Load sentence transformer model for embeddings
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Prepare documents, embeddings, metadata, and IDs
documents = []
embeddings = []
metadatas = []
ids = []

for text in data:
    documents.append(text)
    embeddings.append(model.encode(text))
    metadatas.append({"metadata": text})
    ids.append(str(uuid.uuid4()))

# Create or get the collection
collection_name = "pet_collection_emb"
try:
    pet_collection_emb = client.get_collection(collection_name)
except:
    pet_collection_emb = client.create_collection(collection_name)

# (Optional) Uncomment if you want to add documents again
# pet_collection_emb.add(
#     documents=documents,
#     embeddings=embeddings,
#     metadatas=metadatas,
#     ids=ids
# )

# Define a method for generating a prompt
def generate_prompt(context, question):
    return f"Context:\n{context}\n\nQuestion: {question}"

# Define the user query
#Ispravi Context
query = """Context: I am writing to express my interest in the QA Intern position at
YouTestMe. As a dedicated student in my final year at the University
of Metropolitan with a passion for software development and
problem-solving, I believe that my skills and eagerness to learn
align well with the requirements for this role. Question:Which programming language she has experience ?"""
# How to change battery? Explane in details.
# How to reduce the battery consumption?
# Which university did she finished?
# How to defectory reset?
# Which position I am applying?
# Which program language she has experience ?
# Which programming language she has experience ?

# Create embedding for the query
input_em = model.encode(query).tolist()

# Retrieve similar documents
results = pet_collection_emb.query(
    query_embeddings=[input_em],
    n_results=2
)

# Join retrieved documents
retrieved_docs = "\n".join(results['documents'][0])

# Generate prompt for the local model
prompt = generate_prompt(retrieved_docs, query)

# Load a lightweight QA model locally
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Tokenize and run inference
inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

with torch.no_grad():
    outputs = qa_model.generate(**inputs, max_length=200)

# Decode the generated answer
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the final answer
print("\nAnswer:")
print(answer)

rouge = RougeMetricsEnglish(1)
precision, recall, fscore = rouge("She has experience in QA internal.",
                                   "She has experience in QA internal and work for 4 years.")
print(precision,recall,fscore)


