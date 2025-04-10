from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader
import uuid
import pprint

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


pet_collection_emb = client.get_collection("pet_collection_emb")
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
pprint.pprint(results)


from pdfquery import pdfquery

import re



#use four space as paragraph delimiter to convert the text into list of paragraphs.
# print (re.split('\s{4,}',text))

# pdf = pdfquery.PDFQuery('customers.pdf')
# pdf.load()
#
#
# #convert the pdf to XML
# pdf.tree.write('customers.xml', pretty_print = True)
# pdf
# reader = pypdf.PdfReader('Cover letter programming YouTestMe.pdf')
# print(len(reader.pages))
# print(reader.pages[0].extract_text())