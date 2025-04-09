from smolagents import Tool
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer, util

import datasets


class GuestInfoRetrieverToolST(Tool):
    name = "guest_info_retriever"
    description = "Retrieves detailed information about gala guests based on their name or relation."
    inputs = {
        "query": {
            "type": "string",
            "description": "The name or relation of the guest you want information about.",
        }
    }
    output_type = "string"

    def __init__(self, documents):
        self.is_initialized = False
        self.model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
        self.documents = documents
        docs = [doc.page_content for doc in self.documents]
        self.embedded_documents = self.model.encode(docs)

    def forward(self, query):
        query_embedding = self.model.encode(query)
        top_match = util.semantic_search(query_embedding, self.embedded_documents)[0][0]
        result_index = top_match['corpus_id']
        return self.documents[result_index]


class GuestInfoRetrieverTool(Tool):
    name = "guest_info_retriever"
    description = "Retrieves detailed information about gala guests based on their name or relation."
    inputs = {
        "query": {
            "type": "string",
            "description": "The name or relation of the guest you want information about.",
        }
    }
    output_type = "string"

    def __init__(self, docs):
        self.is_initialized = False
        self.retriever = BM25Retriever.from_documents(docs)

    def forward(self, query: str):
        results = self.retriever.get_relevant_documents(query)
        if results:
            return "\n\n".join([doc.page_content for doc in results[:3]])
        else:
            return "No matching guest information found."


def load_guest_dataset():
    # Load the dataset
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

    return GuestInfoRetrieverTool(docs)
