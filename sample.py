import datasets
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer, util

guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into Document objects
docs = [
    Document(
        page_content="\n".join(
            [
                f"Name: {guest['name']}",
                f"Relation: {guest['relation']}",
                f"Description: {guest['description']}",
                f"Email: {guest['email']}",
            ]
        ),
        metadata={"name": guest["name"]},
    )
    for guest in guest_dataset
]

model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
docss = [doc.page_content for doc in docs]
embedded_documents = model.encode(docss)

query = "Ada Lovelace?"
query_embedding = model.encode(query)
top_match = util.semantic_search(query_embedding, embedded_documents, top_k=3)[0][0]
# result_index = top_match['corpus_id']
if results:
    print("\n\n".join([doc.page_content for doc in results[:3]]))
else:
    print("No matching guest information found.")

print(docs[result_index])
# print(model.encode(docss))

# print(model.encode("Ada Lovelace"))

# docs = [
#     Document(
#         page_content="\n".join(
#             [
#                 f"Name: {guest['name']}",
#                 f"Relation: {guest['relation']}",
#                 f"Description: {guest['description']}",
#                 f"Email: {guest['email']}",
#             ]
#         ),
#         metadata={"name": guest["name"]},
#     )
#     for guest in guest_dataset
# ]

# print(docs[0].page_content)
