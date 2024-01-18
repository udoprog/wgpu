use std::fmt;
use std::marker::PhantomData;
use std::mem::take;

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
    #[error("Error constructing bind group layout")]
    CreateBindGroupLayoutError(
        #[from]
        #[source]
        crate::binding_model::CreateBindGroupLayoutError,
    ),
    #[error("Error creating pipeline layout")]
    CreatePipelineLayoutError(
        #[from]
        #[source]
        crate::binding_model::CreatePipelineLayoutError,
    ),
    #[error("Error creating render pipeline")]
    CreateRenderPipelineError(
        #[from]
        #[source]
        crate::pipeline::CreateRenderPipelineError,
    ),
    #[error("Error creating compute pipeline")]
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

#[must_use = "Must be used with Context::leave"]
pub(crate) struct Enter {
    trace: usize,
    capture: Option<usize>,
}

/// The error returned by wgpu-core functions. This doesn't actually contain any
/// diagnostics, but we use it as proof that error handling has been performed
/// in a function call since `ErrorInner` is only visible inside of this module.
#[non_exhaustive]
pub struct Error;

#[derive(Clone, Copy)]
enum TraceStep {
    /// A field and optionally an associated index of that field.
    Field(&'static str, Option<usize>),
}

/// The traced path of a reported diagnostic.
pub struct DiagnosticTrace<'a> {
    steps: &'a [TraceStep],
}

impl fmt::Display for DiagnosticTrace<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;

        for step in self.steps {
            let first = take(&mut first);

            if !first {
                write!(f, ".")?;
            }

            match *step {
                TraceStep::Field(field, Some(index)) => {
                    write!(f, "{}[{}]", field, index)?;
                }
                TraceStep::Field(field, None) => {
                    write!(f, "{}", field)?;
                }
            }
        }

        Ok(())
    }
}

struct StoredDiagnostic {
    /// The trace the captured diagnostic corresponds to.
    trace: usize,
    /// The captured diagnostic.
    diagnostic: Diagnostic,
}

/// A context capable of tracing and collecting errors raised in wgpu-core.
pub struct Context<'a> {
    /// Captured diagnostics.
    diagnostics: Vec<StoredDiagnostic>,
    /// The current trace.
    trace: Vec<TraceStep>,
    /// Indicates the index of the current captured trace.
    stored_trace: Option<usize>,
    /// A captured trace.
    traces: Vec<Box<[TraceStep]>>,
    /// We hold onto a lifetime because we might want to use it in the future,
    /// to for example plug in a custom tracing implementation. This ensures
    /// that users of Context doesn't misuse it.
    _marker: PhantomData<&'a ()>,
}

impl<'a> Context<'a> {
    /// Construct a new empty context.
    pub const fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
            trace: Vec::new(),
            stored_trace: None,
            traces: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Indicate if we have diagnostics to report.
    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }

    /// Drain all diagnostics from the context..
    pub fn drain(
        &mut self,
    ) -> impl IntoIterator<Item = (Option<DiagnosticTrace<'_>>, Diagnostic)> + '_ {
        self.diagnostics.drain(..).map(|captured| {
            let trace = self
                .traces
                .get(captured.trace)
                .map(|steps| DiagnosticTrace { steps });
            (trace, captured.diagnostic)
        })
    }

    /// Iterate over all diagnostics in the context.
    pub fn iter(&mut self) -> impl IntoIterator<Item = &Diagnostic> + '_ {
        self.diagnostics.iter().map(|captured| &captured.diagnostic)
    }

    /// Enter a field for tracing purposes.
    #[inline]
    pub(crate) fn enter(&mut self, field: &'static str) -> Enter {
        let expected = self.trace.len();
        self.trace.push(TraceStep::Field(field, None));
        self.stored_trace = None;

        Enter {
            trace: expected,
            capture: self.stored_trace.take(),
        }
    }

    /// Leave a field for tracing purposes.
    #[inline]
    pub(crate) fn leave(&mut self, enter: Enter) {
        let fragment = self.trace.pop();
        debug_assert!(fragment.is_some(), "Unbalanced trace");
        debug_assert_eq!(enter.trace, self.trace.len(), "Unbalanced trace");
        self.stored_trace = enter.capture;
    }

    /// Enter the index of a field for tracing purposes.
    ///
    /// This assumes that the index is part of the previously entered field and
    /// does not have to be matched with a corresponding call to `leave`.
    #[inline]
    pub(crate) fn index(&mut self, index: usize) {
        if let Some(TraceStep::Field(_, existing)) = self.trace.last_mut() {
            *existing = Some(index);
        }

        self.stored_trace = None;
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
                self.capture(error.into());
                Err(Error)
            }
        }
    }

    /// Report a diagnostic directly returning the special `Error` marker.
    pub(crate) fn report(&mut self, error: impl Into<Diagnostic>) -> Error {
        self.capture(error.into());
        Error
    }

    /// Capture a diagnostic.
    pub(crate) fn capture(&mut self, diagnostic: Diagnostic) {
        let trace = match self.stored_trace {
            Some(trace) => trace,
            None => {
                let trace = self.traces.len();
                self.traces.push(self.trace.as_slice().into());
                self.stored_trace = Some(trace);
                trace
            }
        };

        self.diagnostics
            .push(StoredDiagnostic { trace, diagnostic });
    }

    /// Internal helper to evaluate a fallible callback, which aids in building
    /// fallible blocks in infallible code which wgpu-core does for *most* of
    /// its API.
    ///
    /// We also ensure that if multiple diagnostics are reported and the block
    /// for some reason forgets to return an error, we do so here instead as a
    /// last resort. For this we also include a debug_assert! to try and ensure
    /// that this is caught during testing instead.
    #[inline(always)]
    pub(crate) fn try_block<F, T>(&mut self, cb: F) -> Result<T, Error>
    where
        F: FnOnce(&mut Self) -> Result<T, Error>,
    {
        let value = cb(self)?;

        debug_assert!(self.is_empty(), "Diagnostics was incorrectly reported");

        if !self.is_empty() {
            log::warn!("Diagnostics were incorrectly reported but recovered");
            return Err(Error);
        }

        Ok(value)
    }
}
