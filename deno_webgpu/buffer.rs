// Copyright 2018-2023 the Deno authors. All rights reserved. MIT license.

use deno_core::error::type_error;
use deno_core::error::AnyError;
use deno_core::futures::channel::oneshot;
use deno_core::op2;
use deno_core::OpState;
use deno_core::Resource;
use deno_core::ResourceId;
use std::borrow::Cow;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;
use wgpu_core::resource::BufferAccessResult;

use super::error::DomExceptionOperationError;
use super::error::WebGpuResult;

pub(crate) struct WebGpuBuffer {
    pub(crate) instance: super::Instance,
    pub(crate) id: wgpu_core::id::BufferId,
    pub(crate) core: &'static wgpu_core::CoreTable,
}

impl Resource for WebGpuBuffer {
    fn name(&self) -> Cow<str> {
        "webGPUBuffer".into()
    }

    fn close(self: Rc<Self>) {
        self.core.buffer_drop(&self.instance, self.id, true);
    }
}

struct WebGpuBufferMapped(*mut u8, usize);
impl Resource for WebGpuBufferMapped {
    fn name(&self) -> Cow<str> {
        "webGPUBufferMapped".into()
    }
}

#[op2]
#[serde]
pub fn op_webgpu_create_buffer(
    state: &mut OpState,
    #[smi] device_rid: ResourceId,
    #[string] label: Cow<str>,
    #[number] size: u64,
    usage: u32,
    mapped_at_creation: bool,
) -> Result<WebGpuResult, AnyError> {
    let instance = state.borrow::<super::Instance>();
    let device = state
        .resource_table
        .get::<super::WebGpuDevice>(device_rid)?;

    let descriptor = wgpu_core::resource::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu_types::BufferUsages::from_bits(usage)
            .ok_or_else(|| type_error("usage is not valid"))?,
        mapped_at_creation,
    };

    gfx_put!(device => instance.device_create_buffer(
    device.id,
    &descriptor,
    ()
  ) => state, WebGpuBuffer)
}

#[op2(async)]
#[serde]
pub async fn op_webgpu_buffer_get_map_async(
    state: Rc<RefCell<OpState>>,
    #[smi] buffer_rid: ResourceId,
    #[smi] device_rid: ResourceId,
    mode: u32,
    #[number] offset: u64,
    #[number] size: u64,
) -> Result<WebGpuResult, AnyError> {
    let (sender, receiver) = oneshot::channel::<BufferAccessResult>();

    let device_id;
    let core;
    {
        let state_ = state.borrow();
        let instance = state_.borrow::<super::Instance>();
        let buffer = state_.resource_table.get::<WebGpuBuffer>(buffer_rid)?;
        let device = state_
            .resource_table
            .get::<super::WebGpuDevice>(device_rid)?;

        (device_id, core) = (device.id, device.core);

        let callback = Box::new(move |status| {
            sender.send(status).unwrap();
        });

        // TODO(lucacasonato): error handling
        let maybe_err = buffer
            .core
            .buffer_map_async(
                instance,
                buffer.id,
                offset..(offset + size),
                wgpu_core::resource::BufferMapOperation {
                    host: match mode {
                        1 => wgpu_core::device::HostMap::Read,
                        2 => wgpu_core::device::HostMap::Write,
                        _ => unreachable!(),
                    },
                    callback: Some(wgpu_core::resource::BufferMapCallback::from_rust(callback)),
                },
            )
            .err();

        if maybe_err.is_some() {
            return Ok(WebGpuResult::maybe_err(maybe_err));
        }
    }

    let done = Rc::new(RefCell::new(false));
    let done_ = done.clone();
    let device_poll_fut = async move {
        while !*done.borrow() {
            {
                let state = state.borrow();
                let instance = state.borrow::<super::Instance>();
                core.device_poll(instance, device_id, wgpu_types::Maintain::wait())
                    .unwrap();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Ok::<(), AnyError>(())
    };

    let receiver_fut = async move {
        receiver.await??;
        let mut done = done_.borrow_mut();
        *done = true;
        Ok::<(), AnyError>(())
    };

    tokio::try_join!(device_poll_fut, receiver_fut)?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_buffer_get_mapped_range(
    state: &mut OpState,
    #[smi] buffer_rid: ResourceId,
    #[number] offset: u64,
    #[number] size: Option<u64>,
    #[buffer] buf: &mut [u8],
) -> Result<WebGpuResult, AnyError> {
    let instance = state.borrow::<super::Instance>();
    let buffer = state.resource_table.get::<WebGpuBuffer>(buffer_rid)?;

    let (slice_pointer, range_size) = buffer
        .core
        .buffer_get_mapped_range(&instance, buffer.id, offset, size)
        .map_err(|e| DomExceptionOperationError::new(&e.to_string()))?;

    let slice = unsafe { std::slice::from_raw_parts_mut(slice_pointer, range_size as usize) };
    buf.copy_from_slice(slice);

    let rid = state
        .resource_table
        .add(WebGpuBufferMapped(slice_pointer, range_size as usize));

    Ok(WebGpuResult::rid(rid))
}

#[op2]
#[serde]
pub fn op_webgpu_buffer_unmap(
    state: &mut OpState,
    #[smi] buffer_rid: ResourceId,
    #[smi] mapped_rid: ResourceId,
    #[buffer] buf: Option<&[u8]>,
) -> Result<WebGpuResult, AnyError> {
    let mapped_resource = state
        .resource_table
        .take::<WebGpuBufferMapped>(mapped_rid)?;
    let instance = state.borrow::<super::Instance>();
    let buffer = state.resource_table.get::<WebGpuBuffer>(buffer_rid)?;

    if let Some(buf) = buf {
        let slice = unsafe { std::slice::from_raw_parts_mut(mapped_resource.0, mapped_resource.1) };
        slice.copy_from_slice(buf);
    }

    gfx_ok!(buffer => instance.buffer_unmap(buffer.id))
}
