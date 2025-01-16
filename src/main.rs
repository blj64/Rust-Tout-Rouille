use rayon::prelude::*; // Rayon est utilisé pour paralléliser les calculs de similarité
use std::collections::HashMap; // HashMap pour gérer les collections et documents
use std::sync::Mutex; // Mutex pour protéger les accès concurrents
use std::cmp::Ordering; // Ordering pour trier les similarités

/// Structure représentant un document, identifié par un `id` unique et associé à un vecteur d'embedding.
#[derive(Clone)]
struct Document {
    id: String,         // Identifiant unique du document
    embedding: Vec<f32>, // Embedding du document (vecteur numérique)
}

/// Structure représentant une collection de documents.
/// Chaque collection est une HashMap avec des `id` comme clés et des `Document` comme valeurs.
#[derive(Default, Clone)]
struct Collection {
    name: String,                      // Nom de la collection
    documents: HashMap<String, Document>, // Documents stockés dans la collection
}

impl Collection {
    /// Crée une nouvelle collection avec un nom donné.
    pub fn new(name: &str) -> Self {
        Collection {
            name: name.to_string(),
            documents: HashMap::new(),
        }
    }

    /// Ajoute un document à la collection.
    pub fn add_document(&mut self, id: &str, embedding: Vec<f32>) {
        self.documents.insert(
            id.to_string(),
            Document {
                id: id.to_string(),
                embedding,
            },
        );
    }

    /// Supprime un document de la collection en utilisant son `id`.
    pub fn remove_document(&mut self, id: &str) {
        self.documents.remove(id);
    }

    /// Retourne les `top_n` documents les plus similaires à un embedding donné.
    pub fn get_similar_documents(&self, query_embedding: &[f32], top_n: usize) -> Vec<(String, f32)> {
        // Calcul parallèle des similarités cosinus entre l'embedding de requête et chaque document.
        let similarities: Vec<(String, f32)> = self
            .documents
            .par_iter()
            .map(|(_, doc)| {
                let similarity = cosine_similarity(&doc.embedding, query_embedding);
                (doc.id.clone(), similarity)
            })
            .collect();

        // Tri des documents par similarité décroissante.
        let mut sorted_similarities = similarities;
        sorted_similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        
        // Retourne les `top_n` documents les plus proches.
        sorted_similarities.into_iter().take(top_n).collect()
    }
}

/// Structure représentant une base de données de collections.
/// Chaque collection est identifiée par un nom unique.
struct Database {
    collections: Mutex<HashMap<String, Collection>>, // Protéger les collections contre les accès concurrents
}

impl Database {
    /// Crée une nouvelle base de données.
    pub fn new() -> Self {
        Database {
            collections: Mutex::new(HashMap::new()),
        }
    }

    /// Crée une nouvelle collection avec un nom donné et l'ajoute à la base.
    pub fn create_collection(&self, name: &str) {
        let mut collections = self.collections.lock().unwrap();
        collections.insert(name.to_string(), Collection::new(name));
    }

    /// Supprime une collection de la base par son nom.
    pub fn delete_collection(&self, name: &str) {
        let mut collections = self.collections.lock().unwrap();
        collections.remove(name);
    }

    /// Récupère une collection par son nom, si elle existe.
    pub fn get_collection(&self, name: &str) -> Option<Collection> {
        let collections = self.collections.lock().unwrap();
        collections.get(name).cloned()
    }
}

/// Calcule la similarité cosinus entre deux vecteurs.
/// Retourne un score entre 0.0 (aucune similarité) et 1.0 (vecteurs identiques).
fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    let dot_product: f32 = vec1.iter().zip(vec2).map(|(a, b)| a * b).sum();
    let magnitude1: f32 = vec1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let magnitude2: f32 = vec2.iter().map(|a| a * a).sum::<f32>().sqrt();

    if magnitude1 > 0.0 && magnitude2 > 0.0 {
        dot_product / (magnitude1 * magnitude2)
    } else {
        0.0 // Retourne 0 si l'un des vecteurs a une norme nulle
    }
}

/// Exemple d'utilisation de la base de données et des collections.
fn main() {
    // Crée une nouvelle base de données.
    let db = Database::new();

    // Crée une collection appelée "test".
    db.create_collection("test");

    // Ajoute des documents à la collection "test".
    if let Some(mut collection) = db.get_collection("test") {
        collection.add_document("doc1", vec![0.1, 0.2, 0.3, 0.4]);
        collection.add_document("caroote", vec![0.2, 0.3, 0.4, 0.5]);
        collection.add_document("doc3", vec![0.9, 0.8, 0.7, 0.6]);

        // Embedding de requête pour trouver des documents similaires.
        let query_embedding = vec![0.2, 0.25, 0.3, 0.35];
        
        // Recherche des 2 documents les plus similaires.
        let results = collection.get_similar_documents(&query_embedding, 2);

        // Affiche les résultats de la recherche.
        println!("Top 2 similar documents:");
        for (id, similarity) in results {
            println!("Document ID: {}, Similarity: {:.4}", id, similarity);
        }
    }
}
