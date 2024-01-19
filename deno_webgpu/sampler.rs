// Copyright 2018-2023 the Deno authors. All rights reserved. MIT license.

use deno_core::error::AnyError;
use deno_core::op2;
use deno_core::OpState;
use deno_core::Resource;
use deno_core::ResourceId;
use serde::Deserialize;
use std::borrow::Cow;
use std::rc::Rc;

use super::error::WebGpuResult;

pub(crate) struct WebGpuSampler {
    pub(crate) instance: crate::Instance,
    pub(crate) id: wgpu_core::id::SamplerId,
    pub(crate) core: &'static wgpu_core::CoreTable,
}

impl Resource for WebGpuSampler {
    fn name(&self) -> Cow<str> {
        "webGPUSampler".into()
    }

    fn close(self: Rc<Self>) {
        self.core.sampler_drop(&self.instance, self.id);
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateSamplerArgs {
    device_rid: ResourceId,
    label: String,
    address_mode_u: wgpu_types::AddressMode,
    address_mode_v: wgpu_types::AddressMode,
    address_mode_w: wgpu_types::AddressMode,
    mag_filter: wgpu_types::FilterMode,
    min_filter: wgpu_types::FilterMode,
    mipmap_filter: wgpu_types::FilterMode, // TODO: GPUMipmapFilterMode
    lod_min_clamp: f32,
    lod_max_clamp: f32,
    compare: Option<wgpu_types::CompareFunction>,
    max_anisotropy: u16,
}

#[op2]
#[serde]
pub fn op_webgpu_create_sampler(
    state: &mut OpState,
    #[serde] args: CreateSamplerArgs,
) -> Result<WebGpuResult, AnyError> {
    let instance = state.borrow::<super::Instance>();
    let device = state
        .resource_table
        .get::<super::WebGpuDevice>(args.device_rid)?;

    let descriptor = wgpu_core::resource::SamplerDescriptor {
        label: Some(Cow::Owned(args.label)),
        address_modes: [
            args.address_mode_u,
            args.address_mode_v,
            args.address_mode_w,
        ],
        mag_filter: args.mag_filter,
        min_filter: args.min_filter,
        mipmap_filter: args.mipmap_filter,
        lod_min_clamp: args.lod_min_clamp,
        lod_max_clamp: args.lod_max_clamp,
        compare: args.compare,
        anisotropy_clamp: args.max_anisotropy,
        border_color: None, // native-only
    };

    gfx_put! {
        device => instance.device_create_sampler(
            device.id,
            &descriptor,
            ()
        ) => state, WebGpuSampler
    }
}
