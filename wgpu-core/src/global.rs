use std::sync::Arc;

use crate::{
    hal_api::HalApi,
    hub::HubReport,
    id::{Id, Marker},
    instance::{Instance, Surface},
    registry::{Registry, RegistryReport},
    resource_log,
    storage::Element,
    traits::{Backend, Backends},
};

#[derive(Debug, PartialEq, Eq)]
pub struct GlobalReport {
    pub surfaces: RegistryReport,
    #[cfg(vulkan)]
    pub vulkan: Option<HubReport>,
    #[cfg(metal)]
    pub metal: Option<HubReport>,
    #[cfg(dx12)]
    pub dx12: Option<HubReport>,
    #[cfg(gles)]
    pub gl: Option<HubReport>,
}

impl GlobalReport {
    pub fn surfaces(&self) -> &RegistryReport {
        &self.surfaces
    }
    pub fn hub_report(&self, backend: wgt::Backend) -> &HubReport {
        match backend {
            #[cfg(vulkan)]
            wgt::Backend::Vulkan => self.vulkan.as_ref().unwrap(),
            #[cfg(metal)]
            wgt::Backend::Metal => self.metal.as_ref().unwrap(),
            #[cfg(dx12)]
            wgt::Backend::Dx12 => self.dx12.as_ref().unwrap(),
            #[cfg(gles)]
            wgt::Backend::Gl => self.gl.as_ref().unwrap(),
            _ => panic!("HubReport is not supported on this backend"),
        }
    }
}

pub struct Global {
    pub instance: Instance,
    pub surfaces: Registry<Surface>,
    pub(crate) backends: Backends,
}

impl Global {
    pub fn new(name: &str, instance_desc: wgt::InstanceDescriptor) -> Self {
        profiling::scope!("Global::new");
        Self {
            instance: Instance::new(name, instance_desc),
            surfaces: Registry::without_backend(),
            backends: Backends::new(),
        }
    }

    /// Get a dynamic reference to the backend referenced by the given
    /// identifier.
    ///
    /// Note: This is a stop-gap API provided to access a dynamic backend, until
    /// wgpu-core has been refactored to no longer require it.
    pub fn backend<T>(&self, id: Id<T>) -> &dyn Backend
    where
        T: Marker,
    {
        match id.backend() {
            #[cfg(vulkan)]
            wgt::Backend::Vulkan => self.backends.vulkan.as_ref(),
            #[cfg(metal)]
            wgt::Backend::Metal => self.backends.metal.as_ref(),
            #[cfg(dx12)]
            wgt::Backend::Dx12 => self.backends.dx12.as_ref(),
            #[cfg(gles)]
            wgt::Backend::Gl => self.backends.gl.as_ref(),
            _ => panic!("Identifier {id:?} is not associated with a supported backend"),
        }
    }

    /// Get an owned dynamic reference to the backend referenced by the given
    /// identifier.
    ///
    /// Note: This is a stop-gap API provided to access a dynamic backend, until
    /// wgpu-core has been refactored to no longer require it.
    pub fn backend_arc<T>(&self, id: Id<T>) -> Arc<dyn Backend>
    where
        T: Marker,
    {
        match id.backend() {
            #[cfg(vulkan)]
            wgt::Backend::Vulkan => self.backends.vulkan.clone(),
            #[cfg(metal)]
            wgt::Backend::Metal => self.backends.metal.clone(),
            #[cfg(dx12)]
            wgt::Backend::Dx12 => self.backends.dx12.clone(),
            #[cfg(gles)]
            wgt::Backend::Gl => self.backends.gl.clone(),
            _ => panic!("Identifier {id:?} is not associated with a supported backend"),
        }
    }

    /// # Safety
    ///
    /// Refer to the creation of wgpu-hal Instance for every backend.
    pub unsafe fn from_hal_instance<A: HalApi>(name: &str, hal_instance: A::Instance) -> Self {
        profiling::scope!("Global::new");
        Self {
            instance: A::create_instance_from_hal(name, hal_instance),
            surfaces: Registry::without_backend(),
            backends: Backends::new(),
        }
    }

    /// # Safety
    ///
    /// - The raw instance handle returned must not be manually destroyed.
    pub unsafe fn instance_as_hal<A: HalApi>(&self) -> Option<&A::Instance> {
        A::instance_as_hal(&self.instance)
    }

    /// # Safety
    ///
    /// - The raw handles obtained from the Instance must not be manually destroyed
    pub unsafe fn from_instance(instance: Instance) -> Self {
        profiling::scope!("Global::new");
        Self {
            instance,
            surfaces: Registry::without_backend(),
            backends: Backends::new(),
        }
    }

    pub fn clear_backend<A: HalApi>(&self, _dummy: ()) {
        let hub = A::hub(self);
        let surfaces_locked = self.surfaces.read();
        // this is used for tests, which keep the adapter
        hub.clear(&surfaces_locked, false);
    }

    pub fn generate_report(&self) -> GlobalReport {
        GlobalReport {
            surfaces: self.surfaces.generate_report(),
            #[cfg(vulkan)]
            vulkan: if self.instance.vulkan.is_some() {
                Some(self.backends.vulkan.hub.generate_report())
            } else {
                None
            },
            #[cfg(metal)]
            metal: if self.instance.metal.is_some() {
                Some(self.backends.metal.hub.generate_report())
            } else {
                None
            },
            #[cfg(dx12)]
            dx12: if self.instance.dx12.is_some() {
                Some(self.backends.dx12.hub.generate_report())
            } else {
                None
            },
            #[cfg(gles)]
            gl: if self.instance.gl.is_some() {
                Some(self.backends.gl.hub.generate_report())
            } else {
                None
            },
        }
    }
}

impl Drop for Global {
    fn drop(&mut self) {
        profiling::scope!("Global::drop");
        resource_log!("Global::drop");
        let mut surfaces_locked = self.surfaces.write();

        // destroy hubs before the instance gets dropped
        #[cfg(vulkan)]
        {
            self.backends.vulkan.hub.clear(&surfaces_locked, true);
        }
        #[cfg(metal)]
        {
            self.hubs.metal.hub.clear(&surfaces_locked, true);
        }
        #[cfg(dx12)]
        {
            self.backends.dx12.hub.clear(&surfaces_locked, true);
        }
        #[cfg(gles)]
        {
            self.backends.gl.hub.clear(&surfaces_locked, true);
        }

        // destroy surfaces
        for element in surfaces_locked.map.drain(..) {
            if let Element::Occupied(arc_surface, _) = element {
                if let Some(surface) = Arc::into_inner(arc_surface) {
                    self.instance.destroy_surface(surface);
                } else {
                    panic!("Surface cannot be destroyed because is still in use");
                }
            }
        }
    }
}

#[cfg(send_sync)]
fn _test_send_sync(global: &Global) {
    fn test_internal<T: Send + Sync>(_: T) {}
    test_internal(global)
}
