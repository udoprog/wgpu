//! Type-erased and object safe API for wgpu-core including stop-gap measures
//! for slowly transitioning away from using identifiers.

use std::fmt;
use std::sync::Arc;

use wgt::WasmNotSendSync;

use crate::binding_model;
use crate::hal_api::HalApi;
use crate::hub::Hub;
use crate::id;

/// The trait for type-erased wgpu-core device.
pub trait Device: 'static + WasmNotSendSync {
    /// Return the backend this device belongs to.
    fn backend(&self) -> wgt::Backend;
}

/// Trait providing dynamic access to a backend.
pub trait Backend: 'static + fmt::Debug + WasmNotSendSync + BackendBindGroupApi {
    /// A stop-gap measure intended to provide access to devices.
    ///
    /// TODO: Deprecate this once we've figured out what to do with player.
    fn device_by_id(&self, id: id::DeviceId) -> Arc<dyn Device>;
}

/// Trait used to perform downcast parameterised over a particular `HalApi`.
pub(crate) trait DowncastArc {
    type Target<A: HalApi>;

    /// Downcast an `Arc<dyn T>` into a `Arc<Self::Target>`.
    ///
    /// The target implementation depends on the trait object being downcasted
    /// from.
    fn downcast_arc<A: HalApi>(self: Arc<Self>) -> Arc<Self::Target<A>>;
}

impl DowncastArc for dyn Device {
    type Target<A: HalApi> = crate::device::Device<A>;

    #[inline]
    fn downcast_arc<A: HalApi>(self: Arc<Self>) -> Arc<Self::Target<A>> {
        assert_eq!(self.backend(), A::VARIANT, "Unexpected backend type");
        unsafe { Arc::from_raw(Arc::into_raw(self) as *const Self::Target<A>) }
    }
}

// This is currently defined as a separate trait, because it makes it cleaner
// diff-wise to implement. At some point we might want to move these functions
// into either the `Backend` or `Device` trait directly.
pub trait BackendBindGroupApi: 'static + WasmNotSendSync {
    fn device_create_bind_group(
        &self,
        device: Arc<dyn Device>,
        desc: &binding_model::BindGroupDescriptor,
        id_in: Option<id::BindGroupId>,
    ) -> (id::BindGroupId, Option<binding_model::CreateBindGroupError>);

    fn bind_group_label(&self, id: id::BindGroupId) -> String;

    fn bind_group_drop(&self, bind_group_id: id::BindGroupId);
}

/// Runtime details associated with a particular backend. Currently this would
/// just be its hub.
pub struct BackendDetails<A: HalApi> {
    pub(crate) hub: Hub<A>,
}

impl<A: HalApi> Backend for BackendDetails<A> {
    fn device_by_id(&self, id: id::DeviceId) -> Arc<dyn Device> {
        self.hub.devices.get(id).expect("Missing device by id")
    }
}

impl<A: HalApi> BackendDetails<A> {
    fn new() -> Self {
        Self { hub: Hub::new() }
    }
}

impl<A: HalApi> fmt::Debug for BackendDetails<A> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Backend").field(&A::VARIANT).finish()
    }
}

/// References to supported or enabled backends.
pub struct Backends {
    #[cfg(vulkan)]
    pub(crate) vulkan: Arc<BackendDetails<hal::api::Vulkan>>,
    #[cfg(metal)]
    pub(crate) metal: Arc<BackendDetails<hal::api::Metal>>,
    #[cfg(dx12)]
    pub(crate) dx12: Arc<BackendDetails<hal::api::Dx12>>,
    #[cfg(gles)]
    pub(crate) gl: Arc<BackendDetails<hal::api::Gles>>,
    #[cfg(all(not(vulkan), not(metal), not(dx12), not(gles)))]
    pub(crate) empty: Arc<BackendDetails<hal::api::Empty>>,
}

impl Backends {
    pub(crate) fn new() -> Self {
        Self {
            #[cfg(vulkan)]
            vulkan: Arc::new(BackendDetails::new()),
            #[cfg(metal)]
            metal: Arc::new(BackendDetails::new()),
            #[cfg(dx12)]
            dx12: Arc::new(BackendDetails::new()),
            #[cfg(gles)]
            gl: Arc::new(BackendDetails::new()),
            #[cfg(all(not(vulkan), not(metal), not(dx12), not(gles)))]
            empty: Arc::new(BackendDetails::new()),
        }
    }
}
