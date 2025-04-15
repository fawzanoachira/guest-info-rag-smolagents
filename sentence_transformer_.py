import datasets
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer, util
from langchain_community.retrievers import BM25Retriever

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
top_match = util.semantic_search(query_embedding, embedded_documents, top_k=3)[0]
# print(top_match)
result_index = [match['corpus_id'] for match in top_match]
# result_index = [Document(page_content=docs[doc].page_content, metadata=docs[doc].metadata) for doc in result_index[:3]]
result_index = [docs[i] for i in result_index]
print(result_index)
# # print(result_index)
# # result_index = top_match['corpus_id']
# if result_index:
#     print("\n\n".join([doc.page_content for doc in result_index[:3]]))
# else:
#     print("No matching guest information found.")



# retriever = BM25Retriever.from_documents(docs)
# results = retriever.get_relevant_documents(query)
# print(results)
# if results:
#     print( "\n\n".join([doc.page_content for doc in results[:3]]))
# else:
#     print( "No matching guest information found.")









# print(docs[result_index])
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
