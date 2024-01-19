use std::borrow::Cow;
use std::fmt;
use std::ops::Range;
use std::path::Path;
use std::ptr;

use serde::{Deserialize, Serialize};

use crate::binding_model;
use crate::command;
use crate::device;
use crate::device::queue;
use crate::global::Global;
use crate::hal_api::HalApi;
use crate::id;
use crate::identity::{IdentityManagerFactory, Input};
use crate::instance;
use crate::pipeline;
use crate::present;
use crate::resource;
use crate::Label;

/// Helper macro to construct the type-erased core table.
macro_rules! core_table {
    (
        $(#[$($meta:meta)*])*
        $ty_vis:vis struct CoreTable<$generic_top:ident: $generic_bound:path> {
            $(
                $(#[doc = $doc:literal])*
                $(#[doc($($doc_meta:meta)*)])*
                $(#[cfg($($cfg:meta)*)])*
                $(#[fn($sig:ident)])?
                $vis:vis fn $name:ident $(<$generic:ident>)? (&$slf:ident, $($arg_name:ident: $arg:ty),* $(,)?) $(-> $ret:ty)?;
            )*
        }
    ) => {
        $(#[$($meta)*])*
        $ty_vis struct CoreTable {
            $ty_vis backend: wgt::Backend,
            $(
                $(#[cfg($($cfg)*)])*
                $name: $($sig)* fn(&Global<IdentityManagerFactory>, $($arg),*) $(-> $ret)*,
            )*
        }

        impl CoreTable {
            /// Construct a collection of backend functions for the specified API.
            $ty_vis fn new<$generic_top: $generic_bound>() -> &'static CoreTable {
                &CoreTable {
                    backend: A::VARIANT,
                    $(
                        $(#[cfg($($cfg)*)])*
                        $name: Global::<IdentityManagerFactory>::$name $(::<$generic>)*,
                    )*
                }
            }

            $(
                #[inline(always)]
                #[allow(unsafe_op_in_unsafe_fn)]
                $(#[doc = $doc])*
                $(#[doc($($doc_meta)*)])*
                $(#[cfg($($cfg)*)])*
                $vis $($sig)* fn $name(&self, global: &Global<IdentityManagerFactory>, $($arg_name: $arg),*) $(-> $ret)* {
                    (self.$name)(global, $($arg_name),*)
                }
            )*
        }
    }
}

type G = IdentityManagerFactory;

core_table! {
    /// Table of functions and static data used to interact with a particular
    /// wgpu-core backend api.
    ///
    /// This is used through a static instance of [`CoreTable`] and exports
    /// type-erased functions to all the relevant [`Global`] functions which
    /// need to be used by wgpu to interact with a particular backend.
    ///
    /// [`CoreTable`] instances can be compared for equality to test if they
    /// refer to the same backend api.
    ///
    /// To use the non-type erased variants, use
    /// [`gfx_select!`](crate::gfx_select!).
    pub struct CoreTable<A: HalApi> {
        pub fn adapter_get_info<A>(
            &self,
            adapter_id: id::AdapterId,
        ) -> Result<wgt::AdapterInfo, instance::InvalidAdapter>;

        pub fn adapter_get_texture_format_features<A>(
            &self,
            adapter_id: id::AdapterId,
            format: wgt::TextureFormat,
        ) -> Result<wgt::TextureFormatFeatures, instance::InvalidAdapter>;

        pub fn adapter_features<A>(
            &self,
            adapter_id: id::AdapterId,
        ) -> Result<wgt::Features, instance::InvalidAdapter>;

        pub fn adapter_limits<A>(
            &self,
            adapter_id: id::AdapterId,
        ) -> Result<wgt::Limits, instance::InvalidAdapter>;

        pub fn adapter_downlevel_capabilities<A>(
            &self,
            adapter_id: id::AdapterId,
        ) -> Result<wgt::DownlevelCapabilities, instance::InvalidAdapter>;

        pub fn adapter_get_presentation_timestamp<A>(
            &self,
            adapter_id: id::AdapterId,
        ) -> Result<wgt::PresentationTimestamp, instance::InvalidAdapter>;

        pub fn adapter_drop<A>(&self, adapter_id: id::AdapterId);

        pub fn surface_get_current_texture<A>(
            &self,
            surface_id: id::SurfaceId,
            texture_id_in: (),
        ) -> Result<present::SurfaceOutput, present::SurfaceError>;

        pub fn surface_present<A>(
            &self,
            surface_id: id::SurfaceId,
        ) -> Result<wgt::SurfaceStatus, present::SurfaceError>;

        pub fn surface_texture_discard<A>(
            &self,
            surface_id: id::SurfaceId,
        ) -> Result<(), present::SurfaceError>;

        pub fn adapter_request_device<A>(
            &self,
            adapter_id: id::AdapterId,
            desc: &device::DeviceDescriptor,
            trace_path: Option<&Path>,
            device_id_in: Input<G, id::DeviceId>,
            queue_id_in: Input<G, id::QueueId>,
        ) -> (
            id::DeviceId,
            id::QueueId,
            Option<instance::RequestDeviceError>,
        );

        pub fn queue_submit<A>(
            &self,
            queue: id::QueueId,
            buffers: &[id::CommandBufferId],
        ) -> Result<queue::WrappedSubmissionIndex, queue::QueueSubmitError>;

        pub fn adapter_is_surface_supported<A>(
            &self,
            adapter_id: id::AdapterId,
            surface_id: id::SurfaceId,
        ) -> Result<bool, instance::IsSurfaceSupportedError>;

        pub fn surface_get_capabilities<A>(
            &self,
            surface_id: id::SurfaceId,
            adapter_id: id::AdapterId,
        ) -> Result<wgt::SurfaceCapabilities, instance::GetSurfaceSupportError>;

        pub fn device_features<A>(
            &self,
            device_id: id::DeviceId,
        ) -> Result<wgt::Features, device::InvalidDevice>;

        pub fn device_limits<A>(
            &self,
            device_id: id::DeviceId,
        ) -> Result<wgt::Limits, device::InvalidDevice>;

        pub fn device_downlevel_properties<A>(
            &self,
            device_id: id::DeviceId,
        ) -> Result<wgt::DownlevelCapabilities, device::InvalidDevice>;

        pub fn device_create_buffer<A>(
            &self,
            device_id: id::DeviceId,
            desc: &resource::BufferDescriptor,
            id_in: Input<G, id::BufferId>,
        ) -> (id::BufferId, Option<resource::CreateBufferError>);

        /// Assign `id_in` an error with the given `label`.
        ///
        /// Ensure that future attempts to use `id_in` as a buffer ID will propagate
        /// the error, following the WebGPU ["contagious invalidity"] style.
        ///
        /// Firefox uses this function to comply strictly with the WebGPU spec,
        /// which requires [`GPUBufferDescriptor`] validation to be generated on the
        /// Device timeline and leave the newly created [`GPUBuffer`] invalid.
        ///
        /// Ideally, we would simply let [`device_create_buffer`] take care of all
        /// of this, but some errors must be detected before we can even construct a
        /// [`wgpu_types::BufferDescriptor`] to give it. For example, the WebGPU API
        /// allows a `GPUBufferDescriptor`'s [`usage`] property to be any WebIDL
        /// `unsigned long` value, but we can't construct a
        /// [`wgpu_types::BufferUsages`] value from values with unassigned bits
        /// set. This means we must validate `usage` before we can call
        /// `device_create_buffer`.
        ///
        /// When that validation fails, we must arrange for the buffer id to be
        /// considered invalid. This method provides the means to do so.
        ///
        /// ["contagious invalidity"]: https://www.w3.org/TR/webgpu/#invalidity
        /// [`GPUBufferDescriptor`]: https://www.w3.org/TR/webgpu/#dictdef-gpubufferdescriptor
        /// [`GPUBuffer`]: https://www.w3.org/TR/webgpu/#gpubuffer
        /// [`wgpu_types::BufferDescriptor`]: wgt::BufferDescriptor
        /// [`device_create_buffer`]: Global::device_create_buffer
        /// [`usage`]: https://www.w3.org/TR/webgpu/#dom-gputexturedescriptor-usage
        /// [`wgpu_types::BufferUsages`]: wgt::BufferUsages
        pub fn create_buffer_error<A>(&self, id_in: (), label: Label);

        pub fn create_render_bundle_error<A>(
            &self,
            id_in: (),
            label: Label,
        );

        /// Assign `id_in` an error with the given `label`.
        ///
        /// See `create_buffer_error` for more context and explaination.
        pub fn create_texture_error<A>(&self, id_in: (), label: Label);

        #[cfg(feature = "replay")]
        pub fn device_wait_for_buffer<A>(
            &self,
            device_id: id::DeviceId,
            buffer_id: id::BufferId,
        ) -> Result<(), device::WaitIdleError>;

        #[doc(hidden)]
        pub fn device_set_buffer_sub_data<A>(
            &self,
            device_id: id::DeviceId,
            buffer_id: id::BufferId,
            offset: wgt::BufferAddress,
            data: &[u8],
        ) -> resource::BufferAccessResult;

        #[doc(hidden)]
        pub fn device_get_buffer_sub_data<A>(
            &self,
            device_id: id::DeviceId,
            buffer_id: id::BufferId,
            offset: wgt::BufferAddress,
            data: &mut [u8],
        ) -> resource::BufferAccessResult;

        pub fn buffer_label<A>(&self, id: id::BufferId) -> String;

        pub fn buffer_destroy<A>(
            &self,
            buffer_id: id::BufferId,
        ) -> Result<(), resource::DestroyError>;

        pub fn buffer_drop<A>(&self, buffer_id: id::BufferId, wait: bool);

        pub fn device_create_texture<A>(
            &self,
            device_id: id::DeviceId,
            desc: &resource::TextureDescriptor,
            id_in: Input<G, id::TextureId>,
        ) -> (id::TextureId, Option<resource::CreateTextureError>);

        pub fn texture_label<A>(&self, id: id::TextureId) -> String;

        pub fn texture_destroy<A>(
            &self,
            texture_id: id::TextureId,
        ) -> Result<(), resource::DestroyError>;

        pub fn texture_drop<A>(&self, texture_id: id::TextureId, wait: bool);

        pub fn texture_create_view<A>(
            &self,
            texture_id: id::TextureId,
            desc: &resource::TextureViewDescriptor,
            id_in: Input<G, id::TextureViewId>,
        ) -> (id::TextureViewId, Option<resource::CreateTextureViewError>);

        pub fn texture_view_label<A>(&self, id: id::TextureViewId) -> String;

        pub fn texture_view_drop<A>(
            &self,
            texture_view_id: id::TextureViewId,
            wait: bool,
        ) -> Result<(), resource::TextureViewDestroyError>;

        pub fn device_create_sampler<A>(
            &self,
            device_id: id::DeviceId,
            desc: &resource::SamplerDescriptor,
            id_in: (),
        ) -> (id::SamplerId, Option<resource::CreateSamplerError>);

        pub fn sampler_label<A>(&self, id: id::SamplerId) -> String;

        pub fn sampler_drop<A>(&self, sampler_id: id::SamplerId);

        pub fn device_create_bind_group_layout<A>(
            &self,
            device_id: id::DeviceId,
            desc: &binding_model::BindGroupLayoutDescriptor,
            id_in: Input<G, id::BindGroupLayoutId>,
        ) -> (
            id::BindGroupLayoutId,
            Option<binding_model::CreateBindGroupLayoutError>,
        );

        pub fn bind_group_layout_label<A>(&self, id: id::BindGroupLayoutId) -> String;

        pub fn bind_group_layout_drop<A>(&self, bind_group_layout_id: id::BindGroupLayoutId);

        pub fn device_create_pipeline_layout<A>(
            &self,
            device_id: id::DeviceId,
            desc: &binding_model::PipelineLayoutDescriptor,
            id_in: Input<G, id::PipelineLayoutId>,
        ) -> (
            id::PipelineLayoutId,
            Option<binding_model::CreatePipelineLayoutError>,
        );

        pub fn pipeline_layout_label<A>(&self, id: id::PipelineLayoutId) -> String;

        pub fn pipeline_layout_drop<A>(&self, pipeline_layout_id: id::PipelineLayoutId);

        pub fn device_create_bind_group<A>(
            &self,
            device_id: id::DeviceId,
            desc: &binding_model::BindGroupDescriptor,
            id_in: Input<G, id::BindGroupId>,
        ) -> (id::BindGroupId, Option<binding_model::CreateBindGroupError>);

        pub fn bind_group_label<A>(&self, id: id::BindGroupId) -> String;

        pub fn bind_group_drop<A>(&self, bind_group_id: id::BindGroupId);

        pub fn device_create_shader_module<A>(
            &self,
            device_id: id::DeviceId,
            desc: &pipeline::ShaderModuleDescriptor,
            source: pipeline::ShaderModuleSource,
            id_in: Input<G, id::ShaderModuleId>,
        ) -> (
            id::ShaderModuleId,
            Option<pipeline::CreateShaderModuleError>,
        );

        /// # Safety
        ///
        /// This function passes SPIR-V binary to the backend as-is and can potentially result in a
        /// driver crash.
        #[fn(unsafe)]
        pub fn device_create_shader_module_spirv<A>(
            &self,
            device_id: id::DeviceId,
            desc: &pipeline::ShaderModuleDescriptor,
            source: Cow<[u32]>,
            id_in: Input<G, id::ShaderModuleId>,
        ) -> (
            id::ShaderModuleId,
            Option<pipeline::CreateShaderModuleError>,
        );

        pub fn shader_module_label<A>(&self, id: id::ShaderModuleId) -> String;

        pub fn shader_module_drop<A>(&self, shader_module_id: id::ShaderModuleId);

        pub fn device_create_command_encoder<A>(
            &self,
            device_id: id::DeviceId,
            desc: &wgt::CommandEncoderDescriptor<Label>,
            id_in: Input<G, id::CommandEncoderId>,
        ) -> (id::CommandEncoderId, Option<device::DeviceError>);

        pub fn command_buffer_label<A>(&self, id: id::CommandBufferId) -> String;

        pub fn command_encoder_drop<A>(&self, command_encoder_id: id::CommandEncoderId);

        pub fn command_buffer_drop<A>(&self, command_buffer_id: id::CommandBufferId);

        pub fn device_create_render_bundle_encoder(
            &self,
            device_id: id::DeviceId,
            desc: &command::RenderBundleEncoderDescriptor,
        ) -> (
            id::RenderBundleEncoderId,
            Option<command::CreateRenderBundleError>,
        );

        pub fn render_bundle_encoder_finish<A>(
            &self,
            bundle_encoder: command::RenderBundleEncoder,
            desc: &command::RenderBundleDescriptor,
            id_in: Input<G, id::RenderBundleId>,
        ) -> (id::RenderBundleId, Option<command::RenderBundleError>);

        pub fn render_bundle_label<A>(&self, id: id::RenderBundleId) -> String;

        pub fn render_bundle_drop<A>(&self, render_bundle_id: id::RenderBundleId);

        pub fn device_create_query_set<A>(
            &self,
            device_id: id::DeviceId,
            desc: &resource::QuerySetDescriptor,
            id_in: Input<G, id::QuerySetId>,
        ) -> (id::QuerySetId, Option<resource::CreateQuerySetError>);

        pub fn query_set_drop<A>(&self, query_set_id: id::QuerySetId);

        pub fn query_set_label<A>(&self, id: id::QuerySetId) -> String;

        pub fn device_create_render_pipeline<A>(
            &self,
            device_id: id::DeviceId,
            desc: &pipeline::RenderPipelineDescriptor,
            id_in: Input<G, id::RenderPipelineId>,
            implicit_pipeline_ids: Option<device::ImplicitPipelineIds<G>>,
        ) -> (
            id::RenderPipelineId,
            Option<pipeline::CreateRenderPipelineError>,
        );

        /// Get an ID of one of the bind group layouts. The ID adds a refcount,
        /// which needs to be released by calling `bind_group_layout_drop`.
        pub fn render_pipeline_get_bind_group_layout<A>(
            &self,
            pipeline_id: id::RenderPipelineId,
            index: u32,
            id_in: Input<G, id::BindGroupLayoutId>,
        ) -> (
            id::BindGroupLayoutId,
            Option<binding_model::GetBindGroupLayoutError>,
        );

        pub fn render_pipeline_label<A>(&self, id: id::RenderPipelineId) -> String;

        pub fn render_pipeline_drop<A>(&self, render_pipeline_id: id::RenderPipelineId);

        pub fn device_create_compute_pipeline<A>(
            &self,
            device_id: id::DeviceId,
            desc: &pipeline::ComputePipelineDescriptor,
            id_in: Input<G, id::ComputePipelineId>,
            implicit_pipeline_ids: Option<device::ImplicitPipelineIds<G>>,
        ) -> (
            id::ComputePipelineId,
            Option<pipeline::CreateComputePipelineError>,
        );

        /// Get an ID of one of the bind group layouts. The ID adds a refcount,
        /// which needs to be released by calling `bind_group_layout_drop`.
        pub fn compute_pipeline_get_bind_group_layout<A>(
            &self,
            pipeline_id: id::ComputePipelineId,
            index: u32,
            id_in: Input<G, id::BindGroupLayoutId>,
        ) -> (
            id::BindGroupLayoutId,
            Option<binding_model::GetBindGroupLayoutError>,
        );

        pub fn compute_pipeline_label<A>(&self, id: id::ComputePipelineId) -> String;

        pub fn compute_pipeline_drop<A>(&self, compute_pipeline_id: id::ComputePipelineId);

        pub fn surface_configure<A>(
            &self,
            surface_id: id::SurfaceId,
            device_id: id::DeviceId,
            config: &wgt::SurfaceConfiguration<Vec<wgt::TextureFormat>>,
        ) -> Option<present::ConfigureSurfaceError>;

        /// Only triange suspected resource IDs. This helps us to avoid ID collisions
        /// upon creating new resources when re-playing a trace.
        #[cfg(feature = "replay")]
        pub fn device_maintain_ids<A>(&self, device_id: id::DeviceId) -> Result<(), device::InvalidDevice>;

        /// Check `device_id` for freeable resources and completed buffer mappings.
        ///
        /// Return `queue_empty` indicating whether there are more queue submissions still in flight.
        pub fn device_poll<A>(
            &self,
            device_id: id::DeviceId,
            maintain: wgt::Maintain<queue::WrappedSubmissionIndex>,
        ) -> Result<bool, device::WaitIdleError>;

        pub fn poll_all_devices(&self, force_wait: bool) -> Result<bool, device::WaitIdleError>;

        pub fn device_label<A>(&self, id: id::DeviceId) -> String;

        pub fn device_start_capture<A>(&self, id: id::DeviceId);

        pub fn device_stop_capture<A>(&self, id: id::DeviceId);

        pub fn device_drop<A>(&self, device_id: id::DeviceId);

        pub fn device_set_device_lost_closure<A>(
            &self,
            device_id: id::DeviceId,
            device_lost_closure: device::DeviceLostClosure,
        );

        pub fn device_destroy<A>(&self, device_id: id::DeviceId);

        pub fn device_mark_lost<A>(&self, device_id: id::DeviceId, message: &str);

        pub fn queue_drop<A>(&self, queue_id: id::QueueId);

        pub fn buffer_map_async<A>(
            &self,
            buffer_id: id::BufferId,
            range: Range<wgt::BufferAddress>,
            op: resource::BufferMapOperation,
        ) -> resource::BufferAccessResult;

        pub fn buffer_get_mapped_range<A>(
            &self,
            buffer_id: id::BufferId,
            offset: wgt::BufferAddress,
            size: Option<wgt::BufferAddress>,
        ) -> Result<(*mut u8, u64), resource::BufferAccessError>;

        pub fn buffer_unmap<A>(&self, buffer_id: id::BufferId) -> resource::BufferAccessResult;

        pub fn command_encoder_copy_buffer_to_buffer<A>(
            &self,
            command_encoder_id: id::CommandEncoderId,
            source: id::BufferId,
            source_offset: wgt::BufferAddress,
            destination: id::BufferId,
            destination_offset: wgt::BufferAddress,
            size: wgt::BufferAddress,
        ) -> Result<(), command::CopyError>;

        pub fn command_encoder_copy_buffer_to_texture<A>(
            &self,
            command_encoder_id: id::CommandEncoderId,
            source: &command::ImageCopyBuffer,
            destination: &command::ImageCopyTexture,
            copy_size: &wgt::Extent3d,
        ) -> Result<(), command::CopyError>;

        pub fn command_encoder_copy_texture_to_buffer<A>(
            &self,
            command_encoder_id: id::CommandEncoderId,
            source: &command::ImageCopyTexture,
            destination: &command::ImageCopyBuffer,
            copy_size: &wgt::Extent3d,
        ) -> Result<(), command::CopyError>;

        pub fn command_encoder_copy_texture_to_texture<A>(
            &self,
            command_encoder_id: id::CommandEncoderId,
            source: &command::ImageCopyTexture,
            destination: &command::ImageCopyTexture,
            copy_size: &wgt::Extent3d,
        ) -> Result<(), command::CopyError>;

        pub fn command_encoder_run_render_pass<A>(
            &self,
            encoder_id: id::CommandEncoderId,
            pass: &command::RenderPass,
        ) -> Result<(), command::RenderPassError>;

        pub fn command_encoder_run_compute_pass<A>(
            &self,
            encoder_id: id::CommandEncoderId,
            pass: &command::ComputePass,
        ) -> Result<(), command::ComputePassError>;

        pub fn command_encoder_finish<A>(
            &self,
            encoder_id: id::CommandEncoderId,
            _desc: &wgt::CommandBufferDescriptor<Label>,
        ) -> (id::CommandBufferId, Option<command::CommandEncoderError>);

        pub fn command_encoder_push_debug_group<A>(
            &self,
            encoder_id: id::CommandEncoderId,
            label: &str,
        ) -> Result<(), command::CommandEncoderError>;

        pub fn command_encoder_insert_debug_marker<A>(
            &self,
            encoder_id: id::CommandEncoderId,
            label: &str,
        ) -> Result<(), command::CommandEncoderError>;

        pub fn command_encoder_pop_debug_group<A>(
            &self,
            encoder_id: id::CommandEncoderId,
        ) -> Result<(), command::CommandEncoderError>;

        pub fn command_encoder_clear_buffer<A>(
            &self,
            command_encoder_id: id::CommandEncoderId,
            dst: id::BufferId,
            offset: wgt::BufferAddress,
            size: Option<wgt::BufferAddress>,
        ) -> Result<(), command::ClearError>;

        pub fn command_encoder_clear_texture<A>(
            &self,
            command_encoder_id: id::CommandEncoderId,
            dst: id::TextureId,
            subresource_range: &wgt::ImageSubresourceRange,
        ) -> Result<(), command::ClearError>;

        pub fn command_encoder_write_timestamp<A>(
            &self,
            command_encoder_id: id::CommandEncoderId,
            query_set_id: id::QuerySetId,
            query_index: u32,
        ) -> Result<(), command::QueryError>;

        pub fn command_encoder_resolve_query_set<A>(
            &self,
            command_encoder_id: id::CommandEncoderId,
            query_set_id: id::QuerySetId,
            start_query: u32,
            query_count: u32,
            destination: id::BufferId,
            destination_offset: wgt::BufferAddress,
        ) -> Result<(), command::QueryError>;

        pub fn queue_write_buffer<A>(
            &self,
            queue_id: id::QueueId,
            buffer_id: id::BufferId,
            buffer_offset: wgt::BufferAddress,
            data: &[u8],
        ) -> Result<(), device::queue::QueueWriteError>;

        pub fn queue_create_staging_buffer<A>(
            &self,
            queue_id: id::QueueId,
            buffer_size: wgt::BufferSize,
            id_in: Input<G, id::StagingBufferId>,
        ) -> Result<(id::StagingBufferId, *mut u8), device::queue::QueueWriteError>;

        pub fn queue_write_staging_buffer<A>(
            &self,
            queue_id: id::QueueId,
            buffer_id: id::BufferId,
            buffer_offset: wgt::BufferAddress,
            staging_buffer_id: id::StagingBufferId,
        ) -> Result<(), device::queue::QueueWriteError>;

        pub fn queue_write_texture<A>(
            &self,
            queue_id: id::QueueId,
            destination: &command::ImageCopyTexture,
            data: &[u8],
            data_layout: &wgt::ImageDataLayout,
            size: &wgt::Extent3d,
        ) -> Result<(), device::queue::QueueWriteError>;

        pub fn queue_get_timestamp_period<A>(
            &self,
            queue_id: id::QueueId,
        ) -> Result<f32, device::queue::InvalidQueue>;

        pub fn queue_on_submitted_work_done<A>(
            &self,
            queue_id: id::QueueId,
            closure: device::queue::SubmittedWorkDoneClosure,
        ) -> Result<(), device::queue::InvalidQueue>;

        pub fn queue_validate_write_buffer<A>(
            &self,
            queue_id: id::QueueId,
            buffer_id: id::BufferId,
            buffer_offset: u64,
            buffer_size: u64,
        ) -> Result<(), device::queue::QueueWriteError>;
    }
}

impl CoreTable {
    /// Construct a collection of backend functions.
    pub fn from_backend(backend: wgt::Backend) -> &'static Self {
        match Self::try_from_backend(backend) {
            Some(table) => table,
            None => panic!("Unsupported backend: {:?}", backend),
        }
    }

    /// Construct a collection of backend functions.
    pub fn try_from_backend(backend: wgt::Backend) -> Option<&'static Self> {
        match backend {
            wgt::Backend::Empty => {
                crate::gfx_try_if_empty!(CoreTable::new::<crate::api::Empty>())
            }
            wgt::Backend::Vulkan => {
                crate::gfx_try_if_vulkan!(CoreTable::new::<crate::api::Vulkan>())
            }
            wgt::Backend::Metal => {
                crate::gfx_try_if_metal!(CoreTable::new::<crate::api::Metal>())
            }
            wgt::Backend::Dx12 => crate::gfx_try_if_dx12!(CoreTable::new::<crate::api::Dx12>()),
            wgt::Backend::Gl => crate::gfx_try_if_gles!(CoreTable::new::<crate::api::Gles>()),
            _ => None,
        }
    }
}

impl fmt::Debug for &'static CoreTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("CoreTable").field(&self.backend).finish()
    }
}

impl PartialEq for CoreTable {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // NB: The only way to acquire a reference to backend functions is through a static pointer.
        ptr::eq(self, other)
    }
}

impl Serialize for CoreTable {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.backend.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for &'static CoreTable {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let backend = wgt::Backend::deserialize(deserializer)?;

        match CoreTable::try_from_backend(backend) {
            Some(core) => Ok(core),
            None => Err(serde::de::Error::custom(format!(
                "Unsupported backend: {}",
                backend.to_str()
            ))),
        }
    }
}
