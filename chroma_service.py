from builtins import print
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Initialisation de ChromaDB
client = chromadb.Client(Settings(
    persist_directory="./chroma_data",  # Chemin pour la persistance des données
    chroma_db_impl="duckdb+parquet"
))

# Initialisation du modèle d'encodage
model = SentenceTransformer('all-MiniLM-L6-v2')

# Création ou récupération de la collection
collection_name = "documentation"
collection = client.get_or_create_collection(name=collection_name)

def vectorize_text(text):
    """Vectoriser un texte en utilisant SentenceTransformer"""
    return model.encode([text])[0]

def add_document(doc_id, text):
    """Ajouter un document vectorisé à la collection"""
    vector = vectorize_text(text)
    collection.add(documents=[text], metadatas=[{"id": doc_id}], ids=[doc_id], embeddings=[vector])

def search(query, top_k=5):
    """Rechercher les documents similaires à une requête"""
    query_vector = vectorize_text(query)
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)
    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChromaDB Service")
    parser.add_argument("--add", nargs=2, metavar=("ID", "TEXT"), help="Ajouter un document")
    parser.add_argument("--search", metavar="QUERY", help="Rechercher un document")

    args = parser.parse_args()

    if args.add:
        doc_id, text = args.add
        add_document(doc_id, text)
        print(f"Document ajouté avec ID: {doc_id}")

    if args.search:
        query = args.search
        results = search(query)
        print("Résultats:")
        for doc, metadata in zip(results['documents'], results['metadatas']):
            print(f"ID: {metadata['id']} | Document: {doc}")
