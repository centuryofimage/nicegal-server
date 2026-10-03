//! Check staged tag sets against public PE image fixtures, without running inference or printing tags.
use anyhow::{Context, Result, ensure};
use nicegal_core::embedding::ImageEmbeddingModel;
use nicegal_core::tags::{Sensitivity, TagSet};

fn main() -> Result<()> {
    for model in ImageEmbeddingModel::ALL.into_iter().filter(|model| {
        matches!(
            model,
            ImageEmbeddingModel::PeCoreB16 | ImageEmbeddingModel::PeCoreL14
        )
    }) {
        ensure!(
            TagSet::supports(model),
            "staged tags unavailable for {model}"
        );
        let directory = model
            .local_directory()
            .context("local PE export required")?;
        let cases: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("image-tests.json"))?)?;
        let tags = TagSet::load(model)?;
        let mut count = 0;
        for case in cases.as_array().context("fixture array")? {
            let vector = case["embedding"]
                .as_array()
                .context("fixture vector")?
                .iter()
                .map(|value| {
                    value
                        .as_f64()
                        .map(|number| number as f32)
                        .context("numeric vector")
                })
                .collect::<Result<Vec<_>>>()?;
            let ranking = tags.rank(&vector, 15, true)?;
            for group in [&ranking.simple, &ranking.subjects, &ranking.vibes] {
                ensure!(group.len() == 15, "unexpected ranking count");
                ensure!(
                    group
                        .iter()
                        .all(|tag| tag.score.is_finite() && tag.similarity.is_finite()),
                    "non-finite scores"
                );
                ensure!(
                    group
                        .iter()
                        .all(|tag| tag.tag.sensitivity != Sensitivity::Blocked),
                    "blocked tag leaked"
                );
                ensure!(
                    group.windows(2).all(|pair| pair[0].score >= pair[1].score),
                    "ranking order"
                );
            }
            count += 1;
        }
        println!(
            "{}",
            serde_json::json!({"model":model.id(),"publicFixtures":count,"rankings":"passed","blockedFilter":"passed"})
        );
    }
    Ok(())
}
