use anyhow::{Context, Result};
use ndarray::{s, Array1, Array2, ArrayView1};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use std::path::{Path, PathBuf};
use std::sync::Once;

const LOCAL_ORT_DSO: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../models/libonnxruntime.dylib"
);

static INIT_ONNX_RUNTIME: Once = Once::new();

pub struct SileroVadModel {
    session: Session,
    context: Array2<f32>,
    chunk_size: usize,
    context_size: usize,
}

impl SileroVadModel {
    pub fn new(model_path: &Path, chunk_size: usize, context_size: usize) -> Result<Self> {
        ensure_onnxruntime_runtime()?;

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to load ONNX model at {}", model_path.display()))?;

        Ok(Self {
            session,
            context: Array2::zeros((1, context_size)),
            chunk_size,
            context_size,
        })
    }

    pub fn process_chunk(&mut self, chunk: &ArrayView1<f32>) -> Result<f32> {
        if chunk.len() != self.chunk_size {
            anyhow::bail!(
                "chunk must be {} samples, got {}",
                self.chunk_size,
                chunk.len()
            );
        }

        let mut input = Array1::<f32>::zeros(self.chunk_size + self.context_size);
        input
            .slice_mut(s![..self.context_size])
            .assign(&self.context.row(0));
        input
            .slice_mut(s![self.context_size..])
            .assign(chunk);

        let tensor = Tensor::from_array(
            ([1, input.len()], input.as_slice().expect("contiguous input").to_vec()),
        )?
        .into_dyn();

        let outputs = self.session.run(vec![("input", tensor)])?;
        let (_, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to read probability output")?;
        if data.is_empty() {
            anyhow::bail!("Probability output empty");
        }
        let prob = data[0];

        // update context with last context_size samples of the chunk
        self.context
            .row_mut(0)
            .assign(&chunk.slice(s![chunk.len() - self.context_size..]));

        Ok(prob)
    }

    pub fn reset(&mut self) {
        self.context.fill(0.0);
    }
}

fn ensure_onnxruntime_runtime() -> Result<()> {
    use std::env;

    let mut init_result: Result<()> = Ok(());
    INIT_ONNX_RUNTIME.call_once(|| {
        let dylib_path = Path::new(LOCAL_ORT_DSO);
        if !dylib_path.exists() {
            init_result = Err(anyhow::anyhow!(
                "ONNX Runtime dylib not found at {}. Copy libonnxruntime.dylib there \
                 or set ORT_DSO_PATH/ORT_DYLIB_PATH manually.",
                LOCAL_ORT_DSO
            ));
            return;
        }

        let abs_path: PathBuf = dylib_path
            .canonicalize()
            .unwrap_or_else(|_| dylib_path.to_path_buf());

        if env::var_os("ORT_DYLIB_PATH").is_none() {
            env::set_var("ORT_DYLIB_PATH", &abs_path);
        }

        if env::var_os("ORT_DSO_PATH").is_none() {
            env::set_var("ORT_DSO_PATH", &abs_path);
        }

        if let Some(lib_dir) = abs_path.parent() {
            let new_dyld = match env::var_os("DYLD_LIBRARY_PATH") {
                Some(existing) if !existing.is_empty() => {
                    let mut combined = lib_dir.display().to_string();
                    combined.push(':');
                    combined.push_str(existing.to_string_lossy().as_ref());
                    combined
                }
                _ => lib_dir.display().to_string(),
            };
            env::set_var("DYLD_LIBRARY_PATH", new_dyld);
        } else {
            init_result = Err(anyhow::anyhow!(
                "Failed to determine libonnxruntime directory for {}",
                abs_path.display()
            ));
            return;
        }

        if env::var_os("ORT_STRATEGY").is_none() {
            env::set_var("ORT_STRATEGY", "system");
        }
    });

    init_result
}
