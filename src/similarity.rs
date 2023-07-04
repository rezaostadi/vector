use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::cmp::{self, Ordering};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
pub enum Distance {
	#[serde(rename = "euclidean")]
	Euclidean,
	#[serde(rename = "cosine")]
	Cosine,
	#[serde(rename = "dot")]
	DotProduct,
}

pub fn get_distance_fn(metric: Distance) -> impl Fn(&[f32], &[f32]) -> f32 {
	match metric {
		Distance::DotProduct => dot_product,
		Distance::Cosine => cosine_similarity,
		Distance::Euclidean => euclidian_distance,
	}
}

fn euclidian_distance(vec_one: &[f32], vec_two: &[f32]) -> f32 {
	let mut result: f32 = 0.0;

	for (i, j) in vec_one.iter().zip(vec_two) {
		result += (i - j).abs().powi(2);
	}

	result.sqrt()
}

fn cosine_similarity(vec_one: &[f32], vec_two: &[f32]) -> f32 {
	let dot_product = dot_product(vec_one, vec_two);
	let magnitude = magnitude(vec_one) * magnitude(vec_two);

	dot_product / magnitude
}

fn dot_product(xs: &[f32], ys: &[f32]) -> f32 {
	let mut result: f32 = 0.0;

	let len = cmp::min(xs.len(), ys.len());
	let xs = &xs[..len];
	let ys = &ys[..len];

	for i in 0..len {
		result += xs[i] * ys[i];
	}

	result
}

fn magnitude(vec: &[f32]) -> f32 {
	// The magnitude of a vector is the sqrt of its own dotproduct
	dot_product(vec, vec).sqrt()
}

pub struct ScoreIndex {
	pub score: f32,
	pub index: usize,
}

impl PartialEq for ScoreIndex {
	fn eq(&self, other: &Self) -> bool {
		self.score.eq(&other.score)
	}
}

impl Eq for ScoreIndex {}

impl PartialOrd for ScoreIndex {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		// The comparison is intentionally reversed here to make the heap a min-heap
		other.score.partial_cmp(&self.score)
	}
}

impl Ord for ScoreIndex {
	fn cmp(&self, other: &Self) -> Ordering {
		self.partial_cmp(other).unwrap_or(Ordering::Equal)
	}
}