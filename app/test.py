from click import prompt
from openai import responses
from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader
import uuid
from openai import OpenAI
from config import OPENAI_API_KEY
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from openai import OpenAI
from rouge_score import rouge_scorer

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)

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

# ROUGE metric
reference_answer = "She finished university at the University of California."  # <-- zameni pravim očekivanim odgovorom
predicted_answer = response.content.strip()

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference_answer, predicted_answer)

print("\n🧪 ROUGE Evaluation:")
for metric, score in scores.items():
    print(f"{metric}: Precision={score.precision:.4f}, Recall={score.recall:.4f}, F1={score.fmeasure:.4f}")


# pprint.pprint(results)


# os.environ["OPENAI_API_KEY"] = getpass.getpass()

# llm = ChatOpenAI(model="gpt-4o")
#
#
#
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(text)
# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
#
# retriever = vectorstore.as_retriever()
#
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", splits),
#         ("human", "{input}"),
#     ]
# )
#
#
# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)
#
# results = rag_chain.invoke({"input": "What was Nike's revenue in 2023?"})
#
# print(results)