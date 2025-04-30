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


directoryPath = r""
reader = PdfReader("C:\\Users\\imakamai\\Desktop\\CoverLetter.pdf")
number_of_pages = len(reader.pages)
page = reader.pages[0]
text = page.extract_text()

data = text.split(".\n")

client = chromadb.PersistentClient(path="D:\\Python\\FastAPI Bridge Project\\chromadb")

model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

check = 0

search_term = ""
check = 0

documents = []
embeddings = []
metadatas = []
ids = []

for text in data:
    documents.append(text)
    embeddings.append(model.encode(text))
    metadatas.append({"metadata": text})
    ids.append(str(uuid.uuid4()))


collection_name = "pet_collection_emb"
try:
    pet_collection_emb = client.get_collection(collection_name)
except:
    pet_collection_emb = client.create_collection(collection_name)



search_term = input("Unesite pojam koji želite da pretražite: ").lower()
found = False

for i, sentence in enumerate(data):
    if search_term in sentence.lower():
        print(f"Pojam je pronađen u rečenici {i + 1}: {sentence.strip()}")
        found = True

if not found:
    print("Nema poklapanja sa zadatim tekstom.")

# fsearch_term = ""
# check = 0
#
# for i in range(number_of_pages):
#     page = reader.pages[i]
#     page_text = page.extract_text()
#     if page_text and search_term.lower() in page_text.lower():
#         print(f"File: CoverLetter.pdf, Page: {i+1}")
#         check += 1
#
# if check == 0:
#     print("No matches found for the given text.")


def generate_prompt(context, question):
    return f"Context:\n{context}\n\nQuestion: {question}"

#Ispravi Context
query = "Which program language she has experience ?"
# How to change battery? Explane in details.
# How to reduce the battery consumption?
# Which university did she finished?
# How to defectory reset?
# Which position I am applying?
# Which program language she has experience ?
# Which programming language she has experience ?
# Context: I am writing to express my interest in the QA Intern position at
# YouTestMe. As a dedicated student in my final year at the University
# of Metropolitan with a passion for software development and
# problem-solving, I believe that my skills and eagerness to learn
# align well with the requirements for this role.

#""" Context: Srpski jezik je standardizovana varijanta srpskohrvatskoga jezika kojom uglavnom govore Srbi. Službeni je jezik u Srbiji, jedan od tri službena jezika Bosne i Hercegovine i suslužbeni u Crnoj Gori i na Kosovu. Priznat je kao manjinski jezik u Hrvatskoj, Severnoj Makedoniji, Rumuniji, Mađarskoj, Slovačkoj i Češkoj.
#Question:Ko govori srpski?"""

input_em = model.encode(query).tolist()

results = pet_collection_emb.query(
    query_embeddings=[input_em],
    n_results=2
)


retrieved_docs = "\n".join(results['documents'][0])

prompt = generate_prompt(retrieved_docs, query)


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")


inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

with torch.no_grad():
    outputs = qa_model.generate(**inputs, max_length=200)


answer = tokenizer.decode(outputs[0], skip_special_tokens=True)


print("\nAnswer:")
print(answer)




