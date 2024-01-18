use std::marker::PhantomData;

/// The kind of an error.
///
/// Note that these do not exist in a hierarchy.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Diagnostic {
    #[error("Device error")]
    DeviceError(
        #[from]
        #[source]
        crate::device::DeviceError,
    ),
    #[error("Create bind group layout error")]
    CreateBindGroupLayoutError(
        #[from]
        #[source]
        crate::binding_model::CreateBindGroupLayoutError,
    ),
    #[error("Create pipeline layout error")]
    CreatePipelineLayoutError(
        #[from]
        #[source]
        crate::binding_model::CreatePipelineLayoutError,
    ),
    #[error("Create render pipeline error")]
    CreateRenderPipelineError(
        #[from]
        #[source]
        crate::pipeline::CreateRenderPipelineError,
    ),
    #[error("Create compute pipeline error")]
    CreateComputePipelineError(
        #[from]
        #[source]
        crate::pipeline::CreateComputePipelineError,
    ),
    #[error("Implicit layout error")]
    ImplicitLayoutError(
        #[from]
        #[source]
        crate::pipeline::ImplicitLayoutError,
    ),
    #[error("Missing features")]
    MissingFeatures(
        #[from]
        #[source]
        crate::device::MissingFeatures,
    ),
    #[error("Missing downlevel flags")]
    MissingDownlevelFlags(
        #[from]
        #[source]
        crate::device::MissingDownlevelFlags,
    ),
    #[error("Stage error")]
    StageError(
        #[from]
        #[source]
        crate::validation::StageError,
    ),
}

/// The error returned by wgpu-core functions. This doesn't actually contain any
/// diagnostics, but we use it as proof that error handling has been performed
/// in a function call since `ErrorInner` is only visible inside of this module.
#[non_exhaustive]
pub struct Error;

/// A context capable of tracing and collecting errors raised in wgpu-core.
pub struct Context<'a> {
    diagnostics: Vec<Diagnostic>,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Context<'a> {
    /// Construct a new empty context.
    pub const fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Indicate if we have diagnostics.
    pub fn is_empty(&self) -> bool {
        !self.diagnostics.is_empty()
    }

    /// Drain all diagnostics from the context..
    pub fn drain(&mut self) -> impl IntoIterator<Item = Diagnostic> + '_ {
        self.diagnostics.drain(..)
    }

    /// Iterate over all diagnostics in the context.
    pub fn iter(&mut self) -> impl IntoIterator<Item = &Diagnostic> + '_ {
        self.diagnostics.iter()
    }

    /// Either return capture and report an error from a `Result<T, E>`, or
    /// returning a result error-mapped to the special `Error` marker.
    pub(crate) fn result<T, E>(&mut self, result: Result<T, E>) -> Result<T, Error>
    where
        Diagnostic: From<E>,
    {
        match result {
            Ok(value) => Ok(value),
            Err(error) => {
                self.diagnostics.push(error.into());
                Err(Error)
            }
        }
    }

    /// Report a diagnostic directly returning the special `Error` marker.
    pub(crate) fn report(&mut self, error: impl Into<Diagnostic>) -> Error {
        self.diagnostics.push(error.into());
        Error
    }
}
