use anyhow::Context;
use lazy_static::lazy_static;
use rayon::prelude::*;
use schemars::JsonSchema;
use std::{
	collections::{BinaryHeap, HashMap},
	fs,
	path::PathBuf,
};

use crate::similarity::{get_cache_attr, get_distance_fn, normalize, Distance, ScoreIndex};

lazy_static! {
	pub static ref STORE_PATH: PathBuf = PathBuf::from("./storage/db");
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("Collection already exists")]
	UniqueViolation,

	#[error("Collection doesn't exist")]
	NotFound,

	#[error("The dimension of the vector doesn't match the dimension of the collection")]
	DimensionMismatch,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Db {
	pub collections: HashMap<String, Collection>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
pub struct SimilarityResult {
	pub score: f32,
	pub embedding: Embedding,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
pub struct Collection {
	/// Dimension of the vectors in the collection
	pub dimension: usize,
	/// Distance metric used for querying
	pub distance: Distance,
	/// Embeddings in the collection
	#[serde(default)]
	pub embeddings: Vec<Embedding>,
}

fn always_true(_: &Embedding) -> bool {
	true
}

impl Collection {
	pub fn get_similarity(
		&self,
		query: &[f32],
		k: usize,
		predicate_fn: Option<&dyn Fn(&Embedding) -> bool>,
		threshold: Option<f32>,
	) -> Vec<SimilarityResult> {
		let threshold = threshold.unwrap_or(-1.0);
		let memo_attr = get_cache_attr(self.distance, query);
		let distance_fn = get_distance_fn(self.distance);

		let predicate_fn = match predicate_fn {
			Some(predicate_fn) => predicate_fn,
			None => &always_true,
		};

		let scores = self
			.embeddings
			.par_iter()
			.enumerate()
			.map(|(index, embedding)| {
				let score = distance_fn(&embedding.vector, query, memo_attr);
				ScoreIndex { score, index }
			})
			.collect::<Vec<_>>();

		let mut heap = BinaryHeap::new();
		for score_index in scores {
			if (heap.len() < k || score_index < *heap.peek().unwrap())
				&& score_index.score >= threshold
				&& predicate_fn(&self.embeddings[score_index.index])
			{
				heap.push(score_index);

				if heap.len() > k {
					heap.pop();
				}
			}
		}

		heap.into_sorted_vec()
			.into_iter()
			.map(|ScoreIndex { score, index }| SimilarityResult {
				score,
				embedding: self.embeddings[index].clone(),
			})
			.collect()
	}
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
pub struct Embedding {
	pub id: String,
	pub vector: Vec<f32>,
	pub metadata: Option<HashMap<String, String>>,
}

impl Db {
	pub fn new() -> Self {
		Self {
			collections: HashMap::new(),
		}
	}

	pub fn create_collection(
		&mut self,
		name: String,
		dimension: usize,
		distance: Distance,
	) -> Result<Collection, Error> {
		if self.collections.contains_key(&name) {
			return Err(Error::UniqueViolation);
		}

		let collection = Collection {
			dimension,
			distance,
			embeddings: Vec::new(),
		};

		self.collections.insert(name, collection.clone());

		Ok(collection)
	}

	pub fn delete_collection(&mut self, name: &str) -> Result<(), Error> {
		if !self.collections.contains_key(name) {
			return Err(Error::NotFound);
		}

		self.collections.remove(name);

		Ok(())
	}

	pub fn insert_into_collection(
		&mut self,
		collection_name: &str,
		mut embedding: Embedding,
	) -> Result<(), Error> {
		let collection = self
			.collections
			.get_mut(collection_name)
			.ok_or(Error::NotFound)?;

		if collection.embeddings.iter().any(|e| e.id == embedding.id) {
			return Err(Error::UniqueViolation);
		}

		if embedding.vector.len() != collection.dimension {
			return Err(Error::DimensionMismatch);
		}

		// Normalize the vector if the distance metric is cosine, so we can use dot product later
		if collection.distance == Distance::Cosine {
			embedding.vector = normalize(&embedding.vector);
		}

		collection.embeddings.push(embedding);

		Ok(())
	}

	pub fn get_collection(&self, name: &str) -> Option<&Collection> {
		self.collections.get(name)
	}

	fn load_from_store() -> anyhow::Result<Self> {
		if !STORE_PATH.exists() {
			tracing::debug!("Creating database store");
			fs::create_dir_all(STORE_PATH.parent().context("Invalid store path")?)?;

			return Ok(Self::new());
		}

		tracing::debug!("Loading database from store");
		let db = fs::read(STORE_PATH.as_path())?;
		Ok(bincode::deserialize(&db[..])?)
	}

	fn save_to_store(&self) -> anyhow::Result<()> {
		let db = bincode::serialize(self)?;

		fs::write(STORE_PATH.as_path(), db)?;

		Ok(())
	}
}

impl Drop for Db {
	fn drop(&mut self) {
		tracing::debug!("Saving database to store");
		self.save_to_store().ok();
	}
}

pub fn from_store() -> anyhow::Result<Db> {
	Db::load_from_store()
}
