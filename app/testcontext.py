import uuid
import torch
import chromadb

from PyPDF2 import PdfReader
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from rougeEnglish import RougeMetricsEnglish

# ============================================
# Read pdf file
# ============================================

reader = PdfReader("C:\\Users\\imakamai\\Desktop\\SM.pdf")
text = ""
for page in reader.pages:
    extracted = page.extract_text()
    if extracted:
        text += extracted + "\n"


# data = text.split(".\n")
data = sent_tokenize(text)

client = chromadb.PersistentClient(path="D:\\Python\\FastAPI Bridge Project\\chromadb")
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')


documents, embeddings, metadatas, ids = [], [], [], []

# ----- Filtering and adding in ChromaDB -----
for sentence in data:
    if len(sentence.strip())< 15:
        continue
    embedding = model.encode(sentence)
    documents.append(sentence)
    embeddings.append(model.encode(sentence))
    metadatas.append({"metadata": sentence})
    ids.append(str(uuid.uuid4()))

collection_name = "pet_collection_emb"
try:
    pet_collection_emb = client.get_collection(collection_name)
except:
    pet_collection_emb = client.create_collection(collection_name)

pet_collection_emb.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print(f"[INFO] {len(documents)} documents added to ChromaDB.")

# ============================================
# Asking question and generate answer
# ============================================
query = input("Ask a question in English: ")


input_em = model.encode(query).tolist()
results = pet_collection_emb.query(query_embeddings=[input_em], n_results=5)


retrieved_docs = "\n".join(results['documents'][0])

def generate_prompt(context, question):
    return f"Context:\n{context}\n\nQuestion: {question}"

prompt = generate_prompt(retrieved_docs, query)


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

with torch.no_grad():
    outputs = qa_model.generate(**inputs, max_length=512, min_length=100, length_penalty=1.0, early_stopping=False)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nAnswer:")
print(answer)

# ============================================
# Rouge metrics
# ============================================

# print(reference)

# rouge = RougeMetricsEnglish(1)
# precision, recall, fscore = rouge(answer,
#                                    "You can turn Bluetooth on or off in Settings > Connections > Bluetooth.")
# print(precision,recall,fscore)

rouge = RougeMetricsEnglish(1)
precision, recall, fscore = rouge("She has experience in QA internal.",
                                   "She has experience in QA internal and work for 4 years.")
print(precision,recall,fscore)
