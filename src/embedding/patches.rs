//! Per-patch image features in the joint text–image space, for visualizing what a CLIP model
//! sees where.
//!
//! CLIP only aligns its pooled token with text, so projecting the last layer's patch tokens gives
//! noisy, often inverted maps. ClearCLIP (Lan et al., ECCV 2024) recovers usable maps without
//! training by recomputing the last block's attention as query–query attention and dropping its
//! residual connection and MLP. [`clearclip_fragment`] adds that branch to an exported Hugging Face
//! CLIP vision graph as one more output, reusing the graph's own weights.
//!
//! A model whose pooled embedding is a plain mean of a spatial feature map needs no new
//! computation: each map position already lies in the embedding space, and
//! [`feature_map_fragment`] only exposes that map as the patch output.
//!
//! The edit is appended to the serialized model instead of re-encoding it. Protobuf merges a
//! repeated embedded message field into the earlier one, so `model ‖ ModelProto { graph: { new
//! nodes, initializers, outputs } }` parses as the original graph with the branch added. Only node
//! headers are read; the weights are skipped without copying.

use anyhow::{Context, Result, bail};

/// Graph output holding unnormalized patch features: `[batch, prefix + patches, dimensions]`
/// for CLIP and DINOv3, or `[batch, rows, columns, dimensions]` for a spatial feature map.
pub(super) const PATCH_OUTPUT: &str = "nicegal_patch_embeds";

const PREFIX: &str = "/nicegal_clearclip/";

/// How a model's patch features are added to its image graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PatchMethod {
    /// A ClearCLIP branch on a Hugging Face CLIP vision export.
    ClearClip,
    /// An existing `[batch, rows, columns, dimensions]` tensor whose spatial mean is the pooled
    /// embedding before normalization.
    FeatureMap(&'static str),
    /// DINOv3's final normalized token sequence: class, four registers, then spatial patches.
    DinoTokens(&'static str),
    /// PE Core's attention pool applied to each final token alone: class, then spatial patches.
    /// It ships only as the export's prebuilt `image_patched.onnx`.
    PeAttnPool,
}

impl PatchMethod {
    /// Serialized protobuf to append to `model` so it also produces [`PATCH_OUTPUT`].
    pub(super) fn fragment(self, model: &[u8]) -> Result<Vec<u8>> {
        match self {
            Self::ClearClip => clearclip_fragment(model),
            Self::FeatureMap(tensor) => feature_map_fragment(model, tensor),
            Self::DinoTokens(tensor) => feature_map_fragment(model, tensor),
            Self::PeAttnPool => bail!("PE patch features ship only as a prebuilt graph"),
        }
    }

    /// The method's name in API responses.
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::ClearClip => "clearclip",
            Self::FeatureMap(_) => "featureMap",
            Self::DinoTokens(_) => "dinoTokens",
            Self::PeAttnPool => "peAttnPool",
        }
    }

    /// Number of nonspatial tokens before a flat patch sequence.
    pub(super) fn prefix_tokens(self) -> usize {
        match self {
            Self::ClearClip | Self::PeAttnPool => 1,
            Self::DinoTokens(_) => 5,
            Self::FeatureMap(_) => 0,
        }
    }
}

/// Unit-length patch vectors for one image, with the source region the grid covers.
#[derive(Debug, Clone)]
pub struct PatchFeatures {
    pub rows: usize,
    pub columns: usize,
    pub dimensions: usize,
    /// Row-major `rows × columns × dimensions`, each patch normalized to unit length.
    pub patches: Vec<f32>,
    /// The pooled embedding computed from the same pixels, as the image index stores it.
    pub embedding: Vec<f32>,
    /// The part of the oriented source image the grid covers, as fractions of its width and
    /// height: `[x, y, width, height]`. Values outside 0 to 1 mean the grid includes padding.
    pub region: [f64; 4],
    /// How the patch vectors were computed: `clearclip` or `featureMap`.
    pub method: &'static str,
}

/// Serialized protobuf to append to a Hugging Face CLIP vision export so it also produces
/// [`PATCH_OUTPUT`]. Fails, rather than guessing, when the graph is not the expected layout.
fn clearclip_fragment(model: &[u8]) -> Result<Vec<u8>> {
    let nodes = graph_nodes(model)?;
    let find = |name: &str, op_type: &str| -> Result<&Node<'_>> {
        let node = nodes
            .iter()
            .find(|node| node.name == name)
            .with_context(|| format!("image graph has no node {name}"))?;
        if node.op_type != op_type {
            bail!("image graph node {name} is {}, not {op_type}", node.op_type);
        }
        Ok(node)
    };
    let last = nodes
        .iter()
        .filter_map(|node| {
            node.name
                .strip_prefix("/vision_model/encoder/layers.")?
                .split('/')
                .next()?
                .parse::<u32>()
                .ok()
        })
        .max()
        .context("image graph has no encoder layers")?;
    let block = format!("/vision_model/encoder/layers.{last}/self_attn");

    // q and k are each pre-scaled by the square root of the attention scale, so the scaled query
    // times its own transpose carries the full scale.
    let query_projection = output(find(&format!("{block}/q_proj/Add"), "Add")?)?;
    let query_reshape = find(&format!("{block}/Reshape"), "Reshape")?;
    let query_heads = find(&format!("{block}/Transpose"), "Transpose")?;
    let scaled_query = find(&format!("{block}/Mul"), "Mul")?;
    find(&format!("{block}/Mul_1"), "Mul")?;
    let value_projection = output(find(&format!("{block}/v_proj/Add"), "Add")?)?;
    let value_reshape = find(&format!("{block}/Reshape_2"), "Reshape")?;
    let value_heads = find(&format!("{block}/Transpose_1"), "Transpose")?;
    if input(query_reshape, 0)? != query_projection
        || input(query_heads, 0)? != output(query_reshape)?
        || input(scaled_query, 0)? != output(query_heads)?
        || input(value_reshape, 0)? != value_projection
        || input(value_heads, 0)? != output(value_reshape)?
    {
        bail!("image graph's last attention block has an unexpected layout");
    }
    let out_matmul = find(&format!("{block}/out_proj/MatMul"), "MatMul")?;
    let out_add = find(&format!("{block}/out_proj/Add"), "Add")?;
    let out_bias = out_add
        .inputs
        .iter()
        .copied()
        .find(|name| *name != output(out_matmul).unwrap_or_default())
        .context("attention output projection has no bias")?;
    let layer_norm = find(
        "/vision_model/post_layernorm/LayerNormalization",
        "LayerNormalization",
    )?;
    let projection = find("/visual_projection/MatMul", "MatMul")?;

    let t = |name: &str| format!("{PREFIX}{name}");
    let mut graph = Vec::new();
    let mut node = |op_type: &str, inputs: &[&str], out: &str, attributes: &[Vec<u8>]| {
        let mut encoded = Vec::new();
        for name in inputs {
            put_bytes(&mut encoded, 1, name.as_bytes());
        }
        put_bytes(&mut encoded, 2, out.as_bytes());
        put_bytes(
            &mut encoded,
            3,
            format!("{PREFIX}{op_type}_{out}").as_bytes(),
        );
        put_bytes(&mut encoded, 4, op_type.as_bytes());
        for attribute in attributes {
            put_bytes(&mut encoded, 5, attribute);
        }
        put_bytes(&mut graph, 1, &encoded);
    };
    let scaled_query = output(scaled_query)?;
    node(
        "Transpose",
        &[scaled_query],
        &t("query_t"),
        &[ints_attribute("perm", &[0, 1, 3, 2])],
    );
    node("MatMul", &[scaled_query, &t("query_t")], &t("logits"), &[]);
    node(
        "Softmax",
        &[&t("logits")],
        &t("attention"),
        &[int_attribute("axis", -1)],
    );
    node(
        "MatMul",
        &[&t("attention"), output(value_heads)?],
        &t("context"),
        &[],
    );
    node(
        "Transpose",
        &[&t("context")],
        &t("context_t"),
        &[ints_attribute("perm", &[0, 2, 1, 3])],
    );
    node(
        "Reshape",
        &[&t("context_t"), &t("merged_shape")],
        &t("merged"),
        &[],
    );
    node(
        "MatMul",
        &[&t("merged"), input(out_matmul, 1)?],
        &t("attended"),
        &[],
    );
    node("Add", &[&t("attended"), out_bias], &t("biased"), &[]);
    let mut norm_inputs = vec![t("biased")];
    norm_inputs.extend(
        layer_norm
            .inputs
            .iter()
            .skip(1)
            .map(|name| (*name).to_owned()),
    );
    node(
        "LayerNormalization",
        &norm_inputs.iter().map(String::as_str).collect::<Vec<_>>(),
        &t("normalized"),
        &layer_norm
            .attributes
            .iter()
            .map(|a| a.to_vec())
            .collect::<Vec<_>>(),
    );
    node(
        "MatMul",
        &[&t("normalized"), input(projection, 1)?],
        PATCH_OUTPUT,
        &[],
    );

    // Reshape [batch, tokens, heads, head width] to [batch, tokens, heads × head width].
    let mut shape = Vec::new();
    put_varint_field(&mut shape, 1, 3);
    put_varint_field(&mut shape, 2, 7); // INT64
    put_bytes(&mut shape, 8, t("merged_shape").as_bytes());
    let values: Vec<u8> = [0_i64, 0, -1]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    put_bytes(&mut shape, 9, &values);
    put_bytes(&mut graph, 5, &shape);

    Ok(with_patch_output(graph))
}

/// Serialized protobuf that names the existing `tensor` as [`PATCH_OUTPUT`].
fn feature_map_fragment(model: &[u8], tensor: &str) -> Result<Vec<u8>> {
    let nodes = graph_nodes(model)?;
    if !nodes.iter().any(|node| node.outputs.contains(&tensor)) {
        bail!("image graph has no tensor {tensor}");
    }
    let mut identity = Vec::new();
    put_bytes(&mut identity, 1, tensor.as_bytes());
    put_bytes(&mut identity, 2, PATCH_OUTPUT.as_bytes());
    put_bytes(&mut identity, 3, format!("{PREFIX}Identity").as_bytes());
    put_bytes(&mut identity, 4, b"Identity");
    let mut graph = Vec::new();
    put_bytes(&mut graph, 1, &identity);
    Ok(with_patch_output(graph))
}

/// `ModelProto { graph }` with [`PATCH_OUTPUT`] declared as a float graph output.
fn with_patch_output(mut graph: Vec<u8>) -> Vec<u8> {
    let mut element = Vec::new();
    put_varint_field(&mut element, 1, 1); // FLOAT
    let mut tensor_type = Vec::new();
    put_bytes(&mut tensor_type, 1, &element);
    let mut value_info = Vec::new();
    put_bytes(&mut value_info, 1, PATCH_OUTPUT.as_bytes());
    put_bytes(&mut value_info, 2, &tensor_type);
    put_bytes(&mut graph, 12, &value_info);

    let mut fragment = Vec::new();
    put_bytes(&mut fragment, 7, &graph);
    fragment
}

/// A patch grid as rows, columns, dimensions, and row-major unit-length patch vectors.
pub(super) type PatchGrid = (usize, usize, usize, Vec<f32>);

/// Normalize one image's [`PATCH_OUTPUT`]: a flat square grid after `prefix_tokens`, or a
/// `[1, rows, columns, dimensions]` spatial feature map when there are no prefix tokens.
pub(super) fn patch_grid(shape: &[usize], data: &[f32], prefix_tokens: usize) -> Result<PatchGrid> {
    let (rows, columns, dimensions, skip) = match (shape, prefix_tokens) {
        (&[1, tokens, dimensions], prefix) if prefix > 0 => {
            let side = (tokens.saturating_sub(prefix) as f64).sqrt().round() as usize;
            if side == 0 || side * side + prefix != tokens {
                bail!(
                    "patch output with {tokens} tokens is not {prefix} prefix tokens and a square grid"
                );
            }
            (side, side, dimensions, prefix * dimensions)
        }
        (&[1, rows, columns, dimensions], 0) => (rows, columns, dimensions, 0),
        _ => bail!("patch output has shape {shape:?}, expected one image"),
    };
    if rows == 0 || columns == 0 || data.len() != skip + rows * columns * dimensions {
        bail!("patch output has {} values for shape {shape:?}", data.len());
    }
    let mut patches = data[skip..].to_vec();
    for patch in patches.chunks_exact_mut(dimensions) {
        let norm = patch
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(f32::EPSILON);
        patch.iter_mut().for_each(|v| *v /= norm);
    }
    Ok((rows, columns, dimensions, patches))
}

/// Split a batched patch output into normalized grids without mixing adjacent images.
pub(super) fn patch_grids(
    shape: &[usize],
    data: &[f32],
    prefix_tokens: usize,
) -> Result<Vec<PatchGrid>> {
    let Some((&batch, single_shape)) = shape.split_first() else {
        bail!("patch output has no batch dimension");
    };
    if batch == 0 || !data.len().is_multiple_of(batch) {
        bail!("patch output has {} values for shape {shape:?}", data.len());
    }
    let values_per_image = data.len() / batch;
    let mut one_shape = Vec::with_capacity(shape.len());
    one_shape.push(1);
    one_shape.extend_from_slice(single_shape);
    data.chunks_exact(values_per_image)
        .map(|image| patch_grid(&one_shape, image, prefix_tokens))
        .collect()
}

#[derive(Debug)]
struct Node<'a> {
    name: &'a str,
    op_type: &'a str,
    inputs: Vec<&'a str>,
    outputs: Vec<&'a str>,
    /// Encoded `AttributeProto`s, copied as-is into new nodes.
    attributes: Vec<&'a [u8]>,
}

fn input<'a>(node: &Node<'a>, index: usize) -> Result<&'a str> {
    node.inputs
        .get(index)
        .copied()
        .with_context(|| format!("image graph node {} lacks input {index}", node.name))
}

fn output<'a>(node: &Node<'a>) -> Result<&'a str> {
    node.outputs
        .first()
        .copied()
        .with_context(|| format!("image graph node {} has no output", node.name))
}

/// `ModelProto.graph.node`, merged across repeated `graph` fields as protobuf parsers do.
fn graph_nodes(model: &[u8]) -> Result<Vec<Node<'_>>> {
    let mut nodes = Vec::new();
    for field in Fields(model) {
        let (number, value) = field?;
        let (7, Value::Bytes(graph)) = (number, value) else {
            continue;
        };
        for field in Fields(graph) {
            let (number, value) = field?;
            let (1, Value::Bytes(encoded)) = (number, value) else {
                continue;
            };
            let mut node = Node {
                name: "",
                op_type: "",
                inputs: Vec::new(),
                outputs: Vec::new(),
                attributes: Vec::new(),
            };
            for field in Fields(encoded) {
                let (number, value) = field?;
                let Value::Bytes(bytes) = value else {
                    continue;
                };
                let text =
                    || std::str::from_utf8(bytes).context("image graph has a non-UTF-8 name");
                match number {
                    1 => node.inputs.push(text()?),
                    2 => node.outputs.push(text()?),
                    3 => node.name = text()?,
                    4 => node.op_type = text()?,
                    5 => node.attributes.push(bytes),
                    _ => {}
                }
            }
            nodes.push(node);
        }
    }
    if nodes.is_empty() {
        bail!("image model has no graph nodes");
    }
    Ok(nodes)
}

enum Value<'a> {
    Scalar,
    Bytes(&'a [u8]),
}

/// Protobuf wire-format fields of one message, skipping values without decoding them.
struct Fields<'a>(&'a [u8]);

impl<'a> Iterator for Fields<'a> {
    type Item = Result<(u64, Value<'a>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0.is_empty() {
            return None;
        }
        let item = (|| {
            let key = take_varint(&mut self.0)?;
            let value = match key & 7 {
                0 => take_varint(&mut self.0).map(|_| Value::Scalar)?,
                1 => take(&mut self.0, 8).map(|_| Value::Scalar)?,
                2 => {
                    let length = usize::try_from(take_varint(&mut self.0)?)?;
                    Value::Bytes(take(&mut self.0, length)?)
                }
                5 => take(&mut self.0, 4).map(|_| Value::Scalar)?,
                wire => bail!("image model has unsupported protobuf wire type {wire}"),
            };
            Ok((key >> 3, value))
        })();
        if item.is_err() {
            self.0 = &[];
        }
        Some(item)
    }
}

fn take<'a>(buffer: &mut &'a [u8], length: usize) -> Result<&'a [u8]> {
    if buffer.len() < length {
        bail!("image model protobuf is truncated");
    }
    let (value, rest) = buffer.split_at(length);
    *buffer = rest;
    Ok(value)
}

fn take_varint(buffer: &mut &[u8]) -> Result<u64> {
    let mut value = 0_u64;
    for shift in (0..64).step_by(7) {
        let [byte, rest @ ..] = *buffer else {
            bail!("image model protobuf is truncated");
        };
        *buffer = rest;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    bail!("image model protobuf has an overlong varint")
}

fn put_varint(buffer: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buffer.push(value as u8 | 0x80);
        value >>= 7;
    }
    buffer.push(value as u8);
}

fn put_varint_field(buffer: &mut Vec<u8>, number: u64, value: u64) {
    put_varint(buffer, number << 3);
    put_varint(buffer, value);
}

fn put_bytes(buffer: &mut Vec<u8>, number: u64, value: &[u8]) {
    put_varint(buffer, (number << 3) | 2);
    put_varint(buffer, value.len() as u64);
    buffer.extend_from_slice(value);
}

fn int_attribute(name: &str, value: i64) -> Vec<u8> {
    let mut attribute = Vec::new();
    put_bytes(&mut attribute, 1, name.as_bytes());
    put_varint_field(&mut attribute, 3, value as u64);
    put_varint_field(&mut attribute, 20, 2); // INT
    attribute
}

fn ints_attribute(name: &str, values: &[i64]) -> Vec<u8> {
    let mut attribute = Vec::new();
    put_bytes(&mut attribute, 1, name.as_bytes());
    for value in values {
        put_varint_field(&mut attribute, 8, *value as u64);
    }
    put_varint_field(&mut attribute, 20, 7); // INTS
    attribute
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(name: &str, op_type: &str, inputs: &[&str], output: &str) -> Vec<u8> {
        let mut encoded = Vec::new();
        for input in inputs {
            put_bytes(&mut encoded, 1, input.as_bytes());
        }
        put_bytes(&mut encoded, 2, output.as_bytes());
        put_bytes(&mut encoded, 3, name.as_bytes());
        put_bytes(&mut encoded, 4, op_type.as_bytes());
        if op_type == "LayerNormalization" {
            put_bytes(&mut encoded, 5, &int_attribute("axis", -1));
        }
        encoded
    }

    /// The node headers ClearCLIP reads, around a weight blob the parser must skip.
    fn clip_like_model(layers: u32) -> Vec<u8> {
        let mut graph = Vec::new();
        put_varint_field(&mut graph, 99, 5);
        put_bytes(&mut graph, 5, &vec![0; 4096]);
        let block = |layer: u32| format!("/vision_model/encoder/layers.{layer}/self_attn");
        for layer in 0..layers {
            let b = block(layer);
            for encoded in [
                node(
                    &format!("{b}/q_proj/Add"),
                    "Add",
                    &["qb", "qm"],
                    &format!("{b}/q"),
                ),
                node(
                    &format!("{b}/Reshape"),
                    "Reshape",
                    &[&format!("{b}/q"), "s"],
                    &format!("{b}/qr"),
                ),
                node(
                    &format!("{b}/Transpose"),
                    "Transpose",
                    &[&format!("{b}/qr")],
                    &format!("{b}/qt"),
                ),
                node(
                    &format!("{b}/Mul"),
                    "Mul",
                    &[&format!("{b}/qt"), "c"],
                    &format!("{b}/qs"),
                ),
                node(
                    &format!("{b}/Mul_1"),
                    "Mul",
                    &["kt", "c"],
                    &format!("{b}/ks"),
                ),
                node(
                    &format!("{b}/v_proj/Add"),
                    "Add",
                    &["vb", "vm"],
                    &format!("{b}/v"),
                ),
                node(
                    &format!("{b}/Reshape_2"),
                    "Reshape",
                    &[&format!("{b}/v"), "s"],
                    &format!("{b}/vr"),
                ),
                node(
                    &format!("{b}/Transpose_1"),
                    "Transpose",
                    &[&format!("{b}/vr")],
                    &format!("{b}/vt"),
                ),
                node(
                    &format!("{b}/out_proj/MatMul"),
                    "MatMul",
                    &["ctx", "Wo"],
                    &format!("{b}/om"),
                ),
                node(
                    &format!("{b}/out_proj/Add"),
                    "Add",
                    &["bo", &format!("{b}/om")],
                    &format!("{b}/o"),
                ),
            ] {
                put_bytes(&mut graph, 1, &encoded);
            }
        }
        put_bytes(
            &mut graph,
            1,
            &node(
                "/vision_model/post_layernorm/LayerNormalization",
                "LayerNormalization",
                &["cls", "g", "b"],
                "ln",
            ),
        );
        put_bytes(
            &mut graph,
            1,
            &node("/visual_projection/MatMul", "MatMul", &["ln", "P"], "proj"),
        );
        let mut model = Vec::new();
        put_varint_field(&mut model, 1, 8);
        put_bytes(&mut model, 7, &graph);
        model
    }

    #[test]
    fn fragment_targets_the_last_block_and_merges_into_the_graph() {
        let mut model = clip_like_model(3);
        let fragment = clearclip_fragment(&model).unwrap();
        model.extend_from_slice(&fragment);
        let nodes = graph_nodes(&model).unwrap();
        let added: Vec<_> = nodes
            .iter()
            .filter(|n| n.name.starts_with(PREFIX))
            .collect();
        assert_eq!(added.len(), 10);
        let block = "/vision_model/encoder/layers.2/self_attn";
        assert_eq!(added[0].inputs, [format!("{block}/qs").as_str()]);
        assert_eq!(added[3].inputs[1], format!("{block}/vt"));
        assert_eq!(added[6].inputs[1], "Wo");
        assert_eq!(added[7].inputs[1], "bo");
        assert_eq!(
            added[8].inputs,
            [&format!("{PREFIX}biased") as &str, "g", "b"]
        );
        assert_eq!(added[8].attributes, [int_attribute("axis", -1).as_slice()]);
        assert_eq!(added[9].inputs[1], "P");
        assert_eq!(added[9].outputs, [PATCH_OUTPUT]);
    }

    #[test]
    fn unexpected_graphs_are_rejected() {
        assert!(clearclip_fragment(&[]).is_err());
        assert!(clearclip_fragment(&[0x3a, 0x05, 0x0a]).is_err());
        let mut model = Vec::new();
        let mut graph = Vec::new();
        put_bytes(
            &mut graph,
            1,
            &node("/visual_projection/MatMul", "MatMul", &["a", "P"], "p"),
        );
        put_bytes(&mut model, 7, &graph);
        assert!(clearclip_fragment(&model).is_err());
    }

    #[test]
    fn patch_grid_drops_the_class_token_and_normalizes() {
        let mut data = vec![9.0, 9.0];
        data.extend([3.0, 4.0, 0.0, 2.0, 1.0, 0.0, 0.0, -5.0]);
        let (rows, columns, dimensions, patches) = patch_grid(&[1, 5, 2], &data, 1).unwrap();
        assert_eq!((rows, columns, dimensions), (2, 2, 2));
        assert_eq!(patches, [0.6, 0.8, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0]);
        assert!(patch_grid(&[1, 4, 2], &data[..8], 1).is_err());
        assert!(patch_grid(&[2, 5, 2], &data, 1).is_err());
    }

    #[test]
    fn patch_grid_reads_a_feature_map_without_a_class_token() {
        let data = [3.0, 4.0, 0.0, 2.0, 1.0, 0.0];
        let (rows, columns, dimensions, patches) = patch_grid(&[1, 1, 3, 2], &data, 0).unwrap();
        assert_eq!((rows, columns, dimensions), (1, 3, 2));
        assert_eq!(patches, [0.6, 0.8, 0.0, 1.0, 1.0, 0.0]);
        assert!(patch_grid(&[1, 6, 2], &data, 0).is_err());
        assert!(patch_grid(&[1, 2, 3, 2], &data, 0).is_err());
    }

    #[test]
    fn dino_grid_drops_class_and_register_tokens() {
        let mut data = vec![99.0; 5 * 2];
        data.extend([3.0, 4.0, 0.0, 2.0, 1.0, 0.0, 0.0, -5.0]);
        let grid = patch_grid(&[1, 9, 2], &data, 5).unwrap();
        assert_eq!((grid.0, grid.1, grid.2), (2, 2, 2));
        assert_eq!(grid.3, [0.6, 0.8, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0]);
        assert!(patch_grid(&[1, 8, 2], &data[..16], 5).is_err());
    }

    #[test]
    fn batch_slicing_keeps_images_separate() {
        let grids =
            patch_grids(&[2, 2, 2], &[99.0, 0.0, 3.0, 4.0, 99.0, 0.0, 0.0, -5.0], 1).unwrap();
        assert_eq!(grids.len(), 2);
        assert_eq!(grids[0].3, [0.6, 0.8]);
        assert_eq!(grids[1].3, [0.0, -1.0]);
        let maps = patch_grids(&[2, 1, 1, 2], &[3.0, 4.0, 0.0, 5.0], 0).unwrap();
        assert_eq!(maps[0].3, [0.6, 0.8]);
        assert_eq!(maps[1].3, [0.0, 1.0]);
    }

    #[test]
    fn feature_map_fragment_names_an_existing_tensor() {
        let mut model = clip_like_model(1);
        assert!(PatchMethod::FeatureMap("missing").fragment(&model).is_err());
        let fragment = PatchMethod::FeatureMap("proj").fragment(&model).unwrap();
        model.extend_from_slice(&fragment);
        let nodes = graph_nodes(&model).unwrap();
        let identity = nodes.iter().find(|n| n.op_type == "Identity").unwrap();
        assert_eq!(
            (identity.inputs.as_slice(), identity.outputs.as_slice()),
            (&["proj"][..], &[PATCH_OUTPUT][..])
        );
    }
}
