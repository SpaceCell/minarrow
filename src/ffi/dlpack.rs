//! # **DLPack FFI** - *Zero-copy tensor interchange with PyTorch, TensorFlow, JAX, and others*
//!
//! Implements the [DLPack](https://github.com/dmlc/dlpack) tensor interchange
//! standard for NdArray, enabling zero-copy data sharing across language and
//! framework boundaries.
//!
//! ## Supported protocols
//! - **DLManagedTensor** (legacy, pre-1.0) - the capsule payload older
//!   consumers expect
//! - **DLManagedTensorVersioned** (DLPack 1.x) - carries the spec version
//!   and the flags word, including the read-only bit
//! - Export: [`export_to_dlpack`] / [`export_to_dlpack_versioned`] wrap an
//!   NdArray, and [`export_view_to_dlpack`] / [`export_view_to_dlpack_versioned`]
//!   wrap an NdArrayV window without copying
//! - Import: [`import_from_dlpack`] / [`import_from_dlpack_versioned`] wrap a
//!   foreign managed tensor as an NdArray
//!
//! ## Version negotiation
//! A Python `__dlpack__(max_version=...)` implementation calls
//! [`export_to_dlpack_versioned`] when the consumer requests major version 1
//! or above, and [`export_to_dlpack`] otherwise. The versioned capsule is
//! named `dltensor_versioned` and the legacy capsule `dltensor`. A consumed
//! capsule renames with a `used_` prefix per the protocol.
//!
//! ## Read-only flag
//! Versioned exports set `DLPACK_FLAG_BITMASK_READ_ONLY` when the backing
//! buffer is shared, since a consumer writing through the pointer would be
//! visible to every other reference. Imported read-only tensors land as
//! shared buffers, so Minarrow's copy-on-write semantics honour the flag -
//! the first mutation copies.
//!
//! ## Layout compatibility
//! NdArray's compact column-major layout is expressed through DLPack's
//! strides field, and the buffer is fully contiguous, so consumers receive
//! a dense tensor with no dead bytes. Column-major is fully supported by
//! PyTorch and NumPy with zero copy. TensorFlow may copy to row-major
//! internally.
//!
//! ## Notes
//! - DLPack uses element strides, matching NdArray's convention
//! - DLPack has no null mask concept - NaN-based missing values align with this
//! - The allocation start is 64-byte aligned, matching the DLPack
//!   recommendation for aligned data pointers
//! - Imported foreign buffers that are not 64-byte aligned copy into an
//!   aligned `Vec64` so the crate's SIMD-alignment invariant holds

use std::ffi::c_void;
use std::sync::Arc;

use crate::enums::error::MinarrowError;
use crate::structs::buffer::Buffer;
use crate::structs::ndarray::NdArray;
use crate::structs::shared_buffer::SharedBuffer;
#[cfg(feature = "views")]
use crate::structs::views::ndarray_view::NdArrayV;
use crate::traits::type_unions::Float;

// ****************************************************************
// DLPack C structs (matching the DLPack header)
// ****************************************************************

/// The DLPack major version this crate produces and understands.
pub const DLPACK_MAJOR_VERSION: u32 = 1;

/// The DLPack minor version this crate produces.
pub const DLPACK_MINOR_VERSION: u32 = 1;

/// Flag bit marking the tensor data as read-only.
pub const DLPACK_FLAG_BITMASK_READ_ONLY: u64 = 1;

/// Flag bit marking the tensor as a copy of the producer's data.
pub const DLPACK_FLAG_BITMASK_IS_COPIED: u64 = 2;

/// Flag bit marking sub-byte element types as byte-padded.
pub const DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED: u64 = 4;

/// Device type code from the DLPack spec.
///
/// Held as a plain integer rather than a Rust enum, since a foreign
/// producer may send any code the spec defines, including ones added
/// after this crate was compiled. Reading an unknown discriminant into
/// a Rust enum would be undefined behaviour.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DLDeviceType(pub i32);

impl DLDeviceType {
    pub const CPU: Self = DLDeviceType(1);
    pub const CUDA: Self = DLDeviceType(2);
    pub const CUDA_HOST: Self = DLDeviceType(3);
    pub const OPENCL: Self = DLDeviceType(4);
    pub const VULKAN: Self = DLDeviceType(7);
    pub const METAL: Self = DLDeviceType(8);
    pub const VPI: Self = DLDeviceType(9);
    pub const ROCM: Self = DLDeviceType(10);
    pub const ROCM_HOST: Self = DLDeviceType(11);
    pub const EXT_DEV: Self = DLDeviceType(12);
    pub const CUDA_MANAGED: Self = DLDeviceType(13);
    pub const ONE_API: Self = DLDeviceType(14);
    pub const WEBGPU: Self = DLDeviceType(15);
    pub const HEXAGON: Self = DLDeviceType(16);
    pub const MAIA: Self = DLDeviceType(17);
}

/// Device descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DLDevice {
    pub device_type: DLDeviceType,
    pub device_id: i32,
}

impl DLDevice {
    /// Host CPU device, id 0. The device every Minarrow buffer lives on,
    /// and the value a Python `__dlpack_device__` implementation returns.
    pub const fn cpu() -> Self {
        DLDevice { device_type: DLDeviceType::CPU, device_id: 0 }
    }
}

/// Data type descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DLDataType {
    /// Type code: 0=int, 1=uint, 2=float, 4=bfloat, 5=complex, 6=bool
    pub code: u8,
    /// Number of bits per element
    pub bits: u8,
    /// Number of SIMD lanes, typically 1
    pub lanes: u16,
}

impl DLDataType {
    /// f32 type descriptor.
    pub const FLOAT32: Self = DLDataType { code: 2, bits: 32, lanes: 1 };

    /// f64 type descriptor.
    pub const FLOAT64: Self = DLDataType { code: 2, bits: 64, lanes: 1 };

    /// Float descriptor for a Minarrow element type.
    pub fn float<T: Float>() -> Self {
        DLDataType { code: 2, bits: (std::mem::size_of::<T>() * 8) as u8, lanes: 1 }
    }
}

/// DLPack spec version carried by the versioned managed tensor.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DLPackVersion {
    pub major: u32,
    pub minor: u32,
}

/// The core DLPack tensor descriptor. Does not own the data.
#[repr(C)]
pub struct DLTensor {
    /// Pointer to the start of the data buffer.
    pub data: *mut c_void,
    /// Device where the data resides.
    pub device: DLDevice,
    /// Number of dimensions.
    pub ndim: i32,
    /// Data type.
    pub dtype: DLDataType,
    /// Shape array with ndim elements.
    pub shape: *mut i64,
    /// Strides array with ndim elements, or null for C-contiguous.
    pub strides: *mut i64,
    /// Byte offset from data pointer to the start of actual data.
    pub byte_offset: u64,
}

/// DLPack legacy managed tensor with ownership semantics.
/// The deleter is called when the consumer is done with the tensor.
#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

unsafe impl Send for DLManagedTensor {}
unsafe impl Sync for DLManagedTensor {}

/// DLPack 1.x managed tensor. Extends the legacy layout with the spec
/// version and a flags word. Field order is ABI - version leads so a
/// consumer can check compatibility before reading the rest.
#[repr(C)]
pub struct DLManagedTensorVersioned {
    pub version: DLPackVersion,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    pub flags: u64,
    pub dl_tensor: DLTensor,
}

unsafe impl Send for DLManagedTensorVersioned {}
unsafe impl Sync for DLManagedTensorVersioned {}

// ****************************************************************
// Export: NdArray / NdArrayV -> managed tensor
// ****************************************************************

/// Keeps the exported source and the i64 shape/strides arrays alive for
/// the lifetime of the managed tensor.
struct DLPackHolder<A> {
    _source: A,
    shape_i64: Vec<i64>,
    strides_i64: Vec<i64>,
}

/// Export an NdArray as a legacy DLPack managed tensor for zero-copy
/// sharing.
///
/// Legacy `DLManagedTensor` carries no read-only flag, so consumers
/// treat the pointer as writable, and a consumer writing through it is
/// visible to every other reference to the buffer. That is the standard
/// DLPack sharing convention across NumPy and PyTorch. Consumers on the
/// 1.x protocol should prefer `to_dlpack_versioned`, which flags shared
/// buffers read-only. A buffer without a stable owned or shared
/// allocation copies into an owned one first, since the consumer holds
/// the base pointer for the tensor's whole lifetime.
///
/// Returns a [`DLPackTensor`] that manages the lifecycle. The NdArray's
/// buffer stays alive through its internal shared reference count. When
/// the owned tensor is dropped, the backing data is released.
///
/// For FFI handoff (e.g. creating a PyCapsule), call `.into_raw()` to
/// transfer ownership to the consumer.
pub fn export_to_dlpack<T: Float>(ndarray: NdArray<T>) -> DLPackTensor {
    let ndarray = if ndarray.data.is_owned() || ndarray.data.is_shared() {
        ndarray
    } else {
        NdArray::from_buffer(ndarray.data.to_owned_copy(), ndarray.shape(), ndarray.strides())
    };
    let shape = ndarray.shape().to_vec();
    let strides = ndarray.strides().to_vec();
    let data_ptr = ndarray.as_slice().as_ptr() as *mut c_void;
    DLPackTensor::from_parts(ndarray, data_ptr, 0, DLDataType::float::<T>(), &shape, &strides)
}

/// Export an NdArray as a DLPack 1.x versioned managed tensor.
///
/// Sets `DLPACK_FLAG_BITMASK_READ_ONLY` when the backing buffer is
/// shared, since a consumer writing through the pointer would be visible
/// to every other reference. A uniquely owned buffer exports writable.
/// A buffer without a stable owned or shared allocation copies into an
/// owned one first, since the consumer holds the base pointer for the
/// tensor's whole lifetime.
pub fn export_to_dlpack_versioned<T: Float>(ndarray: NdArray<T>) -> DLPackTensorVersioned {
    let ndarray = if ndarray.data.is_owned() || ndarray.data.is_shared() {
        ndarray
    } else {
        NdArray::from_buffer(ndarray.data.to_owned_copy(), ndarray.shape(), ndarray.strides())
    };
    let read_only = Arc::strong_count(&ndarray.data) > 1 || ndarray.data.is_shared();
    let flags = if read_only { DLPACK_FLAG_BITMASK_READ_ONLY } else { 0 };
    let shape = ndarray.shape().to_vec();
    let strides = ndarray.strides().to_vec();
    let data_ptr = ndarray.as_slice().as_ptr() as *mut c_void;
    DLPackTensorVersioned::from_parts(
        ndarray, data_ptr, 0, DLDataType::float::<T>(), &shape, &strides, flags,
    )
}

/// Export an NdArrayV window as a legacy DLPack managed tensor without
/// copying. The window offset carries through DLPack's `byte_offset`
/// field, and the view strides carry as element strides, so sliced,
/// transposed, and permuted views all hand over zero-copy.
///
/// Legacy `DLManagedTensor` carries no read-only flag, so a consumer
/// writing through the pointer is visible to every other reference to
/// the backing buffer, per the standard DLPack sharing convention.
/// Consumers on the 1.x protocol should prefer `to_dlpack_versioned`,
/// which flags shared buffers read-only. A backing without a stable
/// owned or shared allocation materialises the window first.
#[cfg(feature = "views")]
pub fn export_view_to_dlpack<T: Float>(view: NdArrayV<T>) -> DLPackTensor {
    if !(view.source.data.is_owned() || view.source.data.is_shared()) {
        return export_to_dlpack(view.to_ndarray());
    }
    let shape = view.shape().to_vec();
    let strides = view.strides().to_vec();
    let byte_offset = (view.offset * std::mem::size_of::<T>()) as u64;
    let data_ptr = view.source.as_slice().as_ptr() as *mut c_void;
    DLPackTensor::from_parts(view, data_ptr, byte_offset, DLDataType::float::<T>(), &shape, &strides)
}

/// Export an NdArrayV window as a DLPack 1.x versioned managed tensor
/// without copying. The window offset carries through DLPack's
/// `byte_offset` field, and the view strides carry as element strides,
/// so sliced, transposed, and permuted views all hand over zero-copy.
/// A view shares its source buffer, so the read-only flag is set
/// whenever another reference to that buffer exists. A backing without
/// a stable owned or shared allocation materialises the window first.
#[cfg(feature = "views")]
pub fn export_view_to_dlpack_versioned<T: Float>(view: NdArrayV<T>) -> DLPackTensorVersioned {
    if !(view.source.data.is_owned() || view.source.data.is_shared()) {
        return export_to_dlpack_versioned(view.to_ndarray());
    }
    let read_only = Arc::strong_count(&view.source.data) > 1 || view.source.data.is_shared();
    let flags = if read_only { DLPACK_FLAG_BITMASK_READ_ONLY } else { 0 };
    let shape = view.shape().to_vec();
    let strides = view.strides().to_vec();
    let byte_offset = (view.offset * std::mem::size_of::<T>()) as u64;
    let data_ptr = view.source.as_slice().as_ptr() as *mut c_void;
    DLPackTensorVersioned::from_parts(
        view, data_ptr, byte_offset, DLDataType::float::<T>(), &shape, &strides, flags,
    )
}

/// Release callback invoked by the foreign consumer when it is done
/// with the tensor. Drops the holder which releases the source.
///
/// # Safety
/// Must only be called once per DLManagedTensor.
unsafe extern "C" fn dlpack_deleter<A>(managed: *mut DLManagedTensor) {
    if managed.is_null() { return; }
    let managed = unsafe { Box::from_raw(managed) };
    if !managed.manager_ctx.is_null() {
        let _holder: Box<DLPackHolder<A>> = unsafe {
            Box::from_raw(managed.manager_ctx as *mut DLPackHolder<A>)
        };
        // holder drops here, releasing the source
    }
}

/// Release callback for the versioned managed tensor.
///
/// # Safety
/// Must only be called once per DLManagedTensorVersioned.
unsafe extern "C" fn dlpack_deleter_versioned<A>(managed: *mut DLManagedTensorVersioned) {
    if managed.is_null() { return; }
    let managed = unsafe { Box::from_raw(managed) };
    if !managed.manager_ctx.is_null() {
        let _holder: Box<DLPackHolder<A>> = unsafe {
            Box::from_raw(managed.manager_ctx as *mut DLPackHolder<A>)
        };
        // holder drops here, releasing the source
    }
}

// ****************************************************************
// DLPackTensor / DLPackTensorVersioned - safe wrappers with Drop
// ****************************************************************

/// Safe wrapper owning a legacy DLPack managed tensor. Calls the deleter
/// on drop.
///
/// This is the return type of `NdArray::to_dlpack()`, analogous to how
/// `to_apache_arrow()` returns an `ArrayRef` that owns its lifecycle.
pub struct DLPackTensor {
    ptr: *mut DLManagedTensor,
}

impl DLPackTensor {
    /// Assemble the managed tensor around a keep-alive source.
    fn from_parts<A>(
        source: A,
        data: *mut c_void,
        byte_offset: u64,
        dtype: DLDataType,
        shape: &[usize],
        strides: &[usize],
    ) -> Self {
        let mut holder = Box::new(DLPackHolder {
            _source: source,
            shape_i64: shape.iter().map(|&s| s as i64).collect(),
            strides_i64: strides.iter().map(|&s| s as i64).collect(),
        });
        let tensor = DLTensor {
            data,
            device: DLDevice::cpu(),
            ndim: shape.len() as i32,
            dtype,
            shape: holder.shape_i64.as_mut_ptr(),
            strides: holder.strides_i64.as_mut_ptr(),
            byte_offset,
        };
        let managed = Box::new(DLManagedTensor {
            dl_tensor: tensor,
            manager_ctx: Box::into_raw(holder) as *mut c_void,
            deleter: Some(dlpack_deleter::<A>),
        });
        DLPackTensor { ptr: Box::into_raw(managed) }
    }

    /// Raw pointer access for FFI consumers that need to take ownership
    /// e.g. when creating a PyCapsule.
    ///
    /// After calling this, the DLPackTensor no longer owns the pointer
    /// and will not call the deleter on drop. The caller is responsible for
    /// ensuring the deleter is called.
    pub fn into_raw(mut self) -> *mut DLManagedTensor {
        let ptr = self.ptr;
        self.ptr = std::ptr::null_mut();
        ptr
    }

    /// Access the underlying DLTensor for reading shape, strides, data.
    pub fn tensor(&self) -> &DLTensor {
        unsafe { &(*self.ptr).dl_tensor }
    }
}

impl Drop for DLPackTensor {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                if let Some(deleter) = (*self.ptr).deleter {
                    deleter(self.ptr);
                }
            }
        }
    }
}

/// Safe wrapper owning a DLPack 1.x versioned managed tensor. Calls the
/// deleter on drop.
///
/// This is the return type of `NdArray::to_dlpack_versioned()`, and the
/// payload a Python `__dlpack__` implementation places in a
/// `dltensor_versioned` capsule via `.into_raw()`.
pub struct DLPackTensorVersioned {
    ptr: *mut DLManagedTensorVersioned,
}

impl DLPackTensorVersioned {
    /// Assemble the versioned managed tensor around a keep-alive source.
    fn from_parts<A>(
        source: A,
        data: *mut c_void,
        byte_offset: u64,
        dtype: DLDataType,
        shape: &[usize],
        strides: &[usize],
        flags: u64,
    ) -> Self {
        let mut holder = Box::new(DLPackHolder {
            _source: source,
            shape_i64: shape.iter().map(|&s| s as i64).collect(),
            strides_i64: strides.iter().map(|&s| s as i64).collect(),
        });
        let tensor = DLTensor {
            data,
            device: DLDevice::cpu(),
            ndim: shape.len() as i32,
            dtype,
            shape: holder.shape_i64.as_mut_ptr(),
            strides: holder.strides_i64.as_mut_ptr(),
            byte_offset,
        };
        let managed = Box::new(DLManagedTensorVersioned {
            version: DLPackVersion {
                major: DLPACK_MAJOR_VERSION,
                minor: DLPACK_MINOR_VERSION,
            },
            manager_ctx: Box::into_raw(holder) as *mut c_void,
            deleter: Some(dlpack_deleter_versioned::<A>),
            flags,
            dl_tensor: tensor,
        });
        DLPackTensorVersioned { ptr: Box::into_raw(managed) }
    }

    /// Raw pointer access for FFI consumers that need to take ownership
    /// e.g. when creating a PyCapsule.
    ///
    /// After calling this, the DLPackTensorVersioned no longer owns the
    /// pointer and will not call the deleter on drop. The caller is
    /// responsible for ensuring the deleter is called.
    pub fn into_raw(mut self) -> *mut DLManagedTensorVersioned {
        let ptr = self.ptr;
        self.ptr = std::ptr::null_mut();
        ptr
    }

    /// Access the underlying DLTensor for reading shape, strides, data.
    pub fn tensor(&self) -> &DLTensor {
        unsafe { &(*self.ptr).dl_tensor }
    }

    /// The DLPack spec version stamped on this tensor.
    pub fn version(&self) -> DLPackVersion {
        unsafe { (*self.ptr).version }
    }

    /// The flags word, including `DLPACK_FLAG_BITMASK_READ_ONLY`.
    pub fn flags(&self) -> u64 {
        unsafe { (*self.ptr).flags }
    }
}

impl Drop for DLPackTensorVersioned {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                if let Some(deleter) = (*self.ptr).deleter {
                    deleter(self.ptr);
                }
            }
        }
    }
}

// ****************************************************************
// Import: managed tensor -> NdArray
// ****************************************************************

/// The foreign managed tensor behind an import, either ABI.
enum ForeignManaged {
    Legacy(*mut DLManagedTensor),
    Versioned(*mut DLManagedTensorVersioned),
}

/// Owns the foreign managed tensor and calls its deleter on drop.
///
/// The DLPack contract makes the deleter responsible for releasing the
/// managed tensor allocation, so the raw pointer is held rather than a
/// `Box`. Wrapping it in a `Box` would free the allocation a second time
/// after the deleter already released it.
struct ForeignDLPack {
    ptr: *const u8,
    len_bytes: usize,
    managed: ForeignManaged,
}

impl AsRef<[u8]> for ForeignDLPack {
    fn as_ref(&self) -> &[u8] {
        if self.len_bytes == 0 || self.ptr.is_null() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len_bytes) }
    }
}

impl Drop for ForeignDLPack {
    fn drop(&mut self) {
        unsafe {
            match self.managed {
                ForeignManaged::Legacy(m) if !m.is_null() => {
                    if let Some(deleter) = (*m).deleter {
                        deleter(m);
                    }
                }
                ForeignManaged::Versioned(m) if !m.is_null() => {
                    if let Some(deleter) = (*m).deleter {
                        deleter(m);
                    }
                }
                _ => {}
            }
        }
        self.managed = ForeignManaged::Legacy(std::ptr::null_mut());
    }
}

unsafe impl Send for ForeignDLPack {}
unsafe impl Sync for ForeignDLPack {}

/// Validation and wrapping pipeline shared by both import entry points.
///
/// The guard owns the foreign tensor, so any validation failure drops it
/// and calls the foreign deleter as the ownership-transfer contract
/// requires. On the success path the guard moves into the SharedBuffer
/// and the deleter runs when the NdArray is dropped.
///
/// # Safety
/// The DLTensor must belong to the managed tensor held by `foreign` and
/// point to accessible CPU memory.
unsafe fn import_dl_tensor<T: Float>(
    tensor: &DLTensor,
    mut foreign: ForeignDLPack,
) -> Result<NdArray<T>, MinarrowError> {
    if tensor.device.device_type != DLDeviceType::CPU {
        return Err(MinarrowError::BridgeError {
            source: "dlpack",
            message: format!(
                "only CPU tensors are supported, got device type {}",
                tensor.device.device_type.0
            ),
        });
    }

    let bits = (std::mem::size_of::<T>() * 8) as u8;
    if tensor.dtype.code != 2 || tensor.dtype.bits != bits || tensor.dtype.lanes != 1 {
        return Err(MinarrowError::BridgeError {
            source: "dlpack",
            message: format!(
                "dtype mismatch, expected code=2 bits={}, got code={} bits={} lanes={}",
                bits, tensor.dtype.code, tensor.dtype.bits, tensor.dtype.lanes
            ),
        });
    }

    if tensor.ndim <= 0 {
        return Err(MinarrowError::BridgeError {
            source: "dlpack",
            message: format!("tensor requires at least one dimension, got ndim={}", tensor.ndim),
        });
    }
    let ndim = tensor.ndim as usize;

    if tensor.shape.is_null() {
        return Err(MinarrowError::BridgeError {
            source: "dlpack",
            message: format!("null shape pointer for ndim={}", ndim),
        });
    }
    let raw_shape = unsafe { std::slice::from_raw_parts(tensor.shape, ndim) };
    if raw_shape.iter().any(|&s| s < 0) {
        return Err(MinarrowError::BridgeError {
            source: "dlpack",
            message: format!("negative dimension in shape {:?}", raw_shape),
        });
    }
    let shape: Vec<usize> = raw_shape.iter().map(|&s| s as usize).collect();

    // Null strides mean C-contiguous i.e. row-major.
    let strides: Vec<usize> = if tensor.strides.is_null() {
        let mut s = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            s[d] = s[d + 1] * shape[d + 1];
        }
        s
    } else {
        let raw_strides = unsafe { std::slice::from_raw_parts(tensor.strides, ndim) };
        if raw_strides.iter().any(|&s| s < 0) {
            return Err(MinarrowError::BridgeError {
                source: "dlpack",
                message: format!(
                    "negative strides {:?} are not representable in NdArray",
                    raw_strides
                ),
            });
        }
        raw_strides.iter().map(|&s| s as usize).collect()
    };

    // Buffer length required to reach the last element, overflow-checked.
    let buf_len = if shape.iter().any(|&s| s == 0) {
        0
    } else {
        let mut max_offset = 0usize;
        for (&s, &st) in shape.iter().zip(strides.iter()) {
            let contribution = (s - 1).checked_mul(st).ok_or_else(|| MinarrowError::BridgeError {
                source: "dlpack",
                message: format!("shape {:?} with strides {:?} overflows the addressable range", shape, strides),
            })?;
            max_offset = max_offset.checked_add(contribution).ok_or_else(|| MinarrowError::BridgeError {
                source: "dlpack",
                message: format!("shape {:?} with strides {:?} overflows the addressable range", shape, strides),
            })?;
        }
        max_offset + 1
    };

    // The byte window the elements span, overflow-checked and capped at
    // isize::MAX per the slice safety contract.
    let len_bytes = buf_len
        .checked_mul(std::mem::size_of::<T>())
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| MinarrowError::BridgeError {
            source: "dlpack",
            message: format!(
                "shape {:?} with strides {:?} spans more than isize::MAX bytes",
                shape, strides
            ),
        })?;

    if tensor.data.is_null() && buf_len > 0 {
        return Err(MinarrowError::BridgeError {
            source: "dlpack",
            message: format!("null data pointer for a tensor of {} elements", buf_len),
        });
    }

    // Data pointer with byte offset, checked for element alignment. The
    // 64-byte SIMD alignment is handled downstream - Buffer::from_shared
    // copies a non-aligned foreign buffer into an aligned Vec64.
    let data_ptr = unsafe { (tensor.data as *const u8).add(tensor.byte_offset as usize) };
    if buf_len > 0 && (data_ptr as usize) % std::mem::align_of::<T>() != 0 {
        return Err(MinarrowError::BridgeError {
            source: "dlpack",
            message: format!("data pointer is not aligned for a {}-bit element", bits),
        });
    }

    // Record the data window now that shape and strides are validated.
    foreign.ptr = data_ptr;
    foreign.len_bytes = len_bytes;

    // Wrap into SharedBuffer -> Buffer -> NdArray. The guard moves into the
    // SharedBuffer, so the deleter now runs when the NdArray is dropped.
    let shared = SharedBuffer::from_owner(foreign);
    let buffer: Buffer<T> = Buffer::from_shared(shared);

    Ok(NdArray::from_buffer(buffer, &shape, &strides))
}

/// Import a foreign legacy DLManagedTensor as a Minarrow NdArray.
///
/// Only CPU float tensors matching the element type `T` are supported. The
/// NdArray takes ownership of the DLManagedTensor and will call its deleter
/// when dropped.
///
/// # Safety
/// - The DLManagedTensor must be valid and point to accessible CPU memory
/// - The caller transfers ownership - the tensor must not be used after this call
/// - The tensor must contain data of element type `T`
/// - The imported NdArray is Send + Sync, so the tensor's deleter must
///   tolerate running on any thread
pub unsafe fn import_from_dlpack<T: Float>(
    managed_ptr: *mut DLManagedTensor,
) -> Result<NdArray<T>, MinarrowError> {
    if managed_ptr.is_null() {
        return Err(MinarrowError::BridgeError {
            source: "dlpack",
            message: "null DLManagedTensor pointer".to_string(),
        });
    }
    let foreign = ForeignDLPack {
        ptr: std::ptr::null(),
        len_bytes: 0,
        managed: ForeignManaged::Legacy(managed_ptr),
    };
    let tensor = unsafe { &(*managed_ptr).dl_tensor };
    unsafe { import_dl_tensor(tensor, foreign) }
}

/// Import a foreign DLPack 1.x versioned managed tensor as a Minarrow
/// NdArray.
///
/// Rejects tensors stamped with a major version newer than
/// [`DLPACK_MAJOR_VERSION`]. A tensor carrying the read-only flag lands
/// as a shared buffer, so the first mutation copies rather than writing
/// through to the producer's memory.
///
/// # Safety
/// - The DLManagedTensorVersioned must be valid and point to accessible CPU memory
/// - The caller transfers ownership - the tensor must not be used after this call
/// - The tensor must contain data of element type `T`
/// - The imported NdArray is Send + Sync, so the tensor's deleter must
///   tolerate running on any thread
pub unsafe fn import_from_dlpack_versioned<T: Float>(
    managed_ptr: *mut DLManagedTensorVersioned,
) -> Result<NdArray<T>, MinarrowError> {
    if managed_ptr.is_null() {
        return Err(MinarrowError::BridgeError {
            source: "dlpack",
            message: "null DLManagedTensorVersioned pointer".to_string(),
        });
    }
    let foreign = ForeignDLPack {
        ptr: std::ptr::null(),
        len_bytes: 0,
        managed: ForeignManaged::Versioned(managed_ptr),
    };
    let version = unsafe { (*managed_ptr).version };
    if version.major > DLPACK_MAJOR_VERSION {
        return Err(MinarrowError::BridgeError {
            source: "dlpack",
            message: format!(
                "DLPack major version {} is newer than the supported {}",
                version.major, DLPACK_MAJOR_VERSION
            ),
        });
    }
    let tensor = unsafe { &(*managed_ptr).dl_tensor };
    unsafe { import_dl_tensor(tensor, foreign) }
}

// ****************************************************************
// Tests
// ****************************************************************

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(all(feature = "views", feature = "select"))]
    use crate::traits::selection::RowSelection;

    #[test]
    fn export_roundtrip_1d() {
        let arr = NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let owned = export_to_dlpack(arr);
        let tensor = owned.tensor();

        assert_eq!(tensor.ndim, 1);
        assert_eq!(tensor.dtype.code, 2);
        assert_eq!(tensor.dtype.bits, 64);
        assert_eq!(tensor.byte_offset, 0);

        let shape = unsafe { std::slice::from_raw_parts(tensor.shape, 1) };
        assert_eq!(shape[0], 3);

        let strides = unsafe { std::slice::from_raw_parts(tensor.strides, 1) };
        assert_eq!(strides[0], 1);

        let data = tensor.data as *const f64;
        assert_eq!(unsafe { *data }, 1.0);
        assert_eq!(unsafe { *data.add(1) }, 2.0);
        assert_eq!(unsafe { *data.add(2) }, 3.0);
        // owned drops here, calling deleter
    }

    #[test]
    fn export_2d_column_major_strides() {
        let arr = NdArray::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]
        );
        let owned = export_to_dlpack(arr);
        let tensor = owned.tensor();

        assert_eq!(tensor.ndim, 2);

        let shape = unsafe { std::slice::from_raw_parts(tensor.shape, 2) };
        assert_eq!(shape[0], 3);
        assert_eq!(shape[1], 2);

        let strides = unsafe { std::slice::from_raw_parts(tensor.strides, 2) };
        assert_eq!(strides[0], 1);
        assert_eq!(strides[1], 3);

        let data = tensor.data as *const f64;
        assert_eq!(unsafe { *data }, 1.0);
        assert_eq!(unsafe { *data.add(2) }, 3.0);
        assert_eq!(unsafe { *data.add(strides[1] as usize) }, 4.0);
        assert_eq!(unsafe { *data.add(strides[1] as usize + 2) }, 6.0);
    }

    #[test]
    fn export_preserves_data_lifetime() {
        let owned = {
            let arr = NdArray::from_slice(&[42.0, 99.0], &[2]);
            export_to_dlpack(arr)
        };
        let data = owned.tensor().data as *const f64;
        assert_eq!(unsafe { *data }, 42.0);
        assert_eq!(unsafe { *data.add(1) }, 99.0);
    }

    #[test]
    fn import_from_exported() {
        let original = NdArray::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]
        );
        let owned = export_to_dlpack(original);
        let raw = owned.into_raw();

        let imported = unsafe { import_from_dlpack::<f64>(raw) }.unwrap();
        assert_eq!(imported.shape(), &[3, 2]);
        assert_eq!(imported.get(&[0, 0]), 1.0);
        assert_eq!(imported.get(&[2, 0]), 3.0);
        assert_eq!(imported.get(&[0, 1]), 4.0);
        assert_eq!(imported.get(&[2, 1]), 6.0);
    }

    #[test]
    fn import_null_ptr_fails() {
        let result = unsafe { import_from_dlpack::<f64>(std::ptr::null_mut()) };
        assert!(result.is_err());
    }

    #[test]
    fn export_3d_strides() {
        let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let arr = NdArray::from_slice(&data, &[2, 3, 4]);
        let owned = export_to_dlpack(arr);
        let tensor = owned.tensor();

        assert_eq!(tensor.ndim, 3);

        let shape = unsafe { std::slice::from_raw_parts(tensor.shape, 3) };
        assert_eq!(shape, &[2, 3, 4]);

        let strides = unsafe { std::slice::from_raw_parts(tensor.strides, 3) };
        assert_eq!(strides[0], 1);
        assert_eq!(strides[1], 2);
        assert_eq!(strides[2], 6);
    }

    #[test]
    fn to_dlpack_method() {
        let arr = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let owned = arr.to_dlpack();
        let tensor = owned.tensor();
        assert_eq!(tensor.ndim, 2);
        let data = tensor.data as *const f64;
        assert_eq!(unsafe { *data }, 1.0);
    }

    #[test]
    fn export_roundtrip_f32() {
        let original = NdArray::<f32>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]
        );
        let owned = export_to_dlpack(original);
        let raw = owned.into_raw();

        let imported = unsafe { import_from_dlpack::<f32>(raw) }.unwrap();
        assert_eq!(imported.shape(), &[3, 2]);
        assert_eq!(imported.get(&[0, 0]), 1.0);
        assert_eq!(imported.get(&[2, 0]), 3.0);
        assert_eq!(imported.get(&[0, 1]), 4.0);
        assert_eq!(imported.get(&[2, 1]), 6.0);
    }

    #[test]
    fn export_f32_dtype_bits() {
        let arr = NdArray::<f32>::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let owned = export_to_dlpack(arr);
        let tensor = owned.tensor();
        assert_eq!(tensor.dtype.code, 2);
        assert_eq!(tensor.dtype.bits, 32);
    }

    #[test]
    fn import_dtype_mismatch_fails() {
        // Export f64, then import as f32 - the bit width mismatch rejects.
        let f64_arr = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let raw = export_to_dlpack(f64_arr).into_raw();
        let result = unsafe { import_from_dlpack::<f32>(raw) };
        assert!(result.is_err());

        // Export f32, then import as f64 - likewise rejected.
        let f32_arr = NdArray::<f32>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let raw = export_to_dlpack(f32_arr).into_raw();
        let result = unsafe { import_from_dlpack::<f64>(raw) };
        assert!(result.is_err());
    }

    // *** Versioned ABI ***********************************************

    #[test]
    fn versioned_roundtrip() {
        let original = NdArray::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]
        );
        let owned = export_to_dlpack_versioned(original);
        assert_eq!(owned.version().major, DLPACK_MAJOR_VERSION);
        assert_eq!(owned.version().minor, DLPACK_MINOR_VERSION);
        assert_eq!(owned.flags(), 0);

        let raw = owned.into_raw();
        let imported = unsafe { import_from_dlpack_versioned::<f64>(raw) }.unwrap();
        assert_eq!(imported.shape(), &[3, 2]);
        assert_eq!(imported.get(&[0, 0]), 1.0);
        assert_eq!(imported.get(&[2, 1]), 6.0);
    }

    #[test]
    fn versioned_read_only_flag_on_shared_buffer() {
        let arr = NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let clone = arr.clone();
        let owned = export_to_dlpack_versioned(clone);
        assert_ne!(owned.flags() & DLPACK_FLAG_BITMASK_READ_ONLY, 0);
        // The original still reads its data after the export drops.
        drop(owned);
        assert_eq!(arr.get(&[0]), 1.0);
    }

    #[test]
    fn versioned_rejects_newer_major() {
        let arr = NdArray::from_slice(&[1.0, 2.0], &[2]);
        let raw = export_to_dlpack_versioned(arr).into_raw();
        unsafe { (*raw).version.major = DLPACK_MAJOR_VERSION + 1 };
        let result = unsafe { import_from_dlpack_versioned::<f64>(raw) };
        assert!(result.is_err());
    }

    // *** View export *************************************************

    #[cfg(all(feature = "views", feature = "select"))]
    #[test]
    fn view_export_carries_byte_offset() {
        let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let arr = NdArray::from_slice(&data, &[5, 2]);
        let view = arr.r(2..5);
        let owned = export_view_to_dlpack(view);
        let tensor = owned.tensor();

        // Zero-copy: the tensor points at the parent's own buffer with
        // the window carried as byte_offset.
        assert_eq!(tensor.byte_offset, 2 * std::mem::size_of::<f64>() as u64);
        assert_eq!(tensor.data as *const u8, arr.as_slice().as_ptr() as *const u8);
        let shape = unsafe { std::slice::from_raw_parts(tensor.shape, 2) };
        assert_eq!(shape, &[3, 2]);

        let raw = owned.into_raw();
        let imported = unsafe { import_from_dlpack::<f64>(raw) }.unwrap();
        assert_eq!(imported.shape(), &[3, 2]);
        assert_eq!(imported.get(&[0, 0]), arr.get(&[2, 0]));
        assert_eq!(imported.get(&[2, 1]), arr.get(&[4, 1]));
    }

    #[cfg(all(feature = "views", feature = "select"))]
    #[test]
    fn view_export_versioned_sets_read_only() {
        let arr = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        // The view shares the source buffer, so the export is read-only.
        let owned = export_view_to_dlpack_versioned(arr.r(0..2));
        assert_ne!(owned.flags() & DLPACK_FLAG_BITMASK_READ_ONLY, 0);
    }

    #[cfg(feature = "views")]
    #[test]
    fn transposed_view_export_roundtrip() {
        let arr = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let raw = export_view_to_dlpack(arr.as_view().transpose()).into_raw();
        let imported = unsafe { import_from_dlpack::<f64>(raw) }.unwrap();
        assert_eq!(imported.shape(), &[2, 3]);
        assert_eq!(imported.get(&[0, 0]), arr.get(&[0, 0]));
        assert_eq!(imported.get(&[1, 2]), arr.get(&[2, 1]));
    }

    // *** Import validation *******************************************

    #[test]
    fn import_rejects_negative_shape() {
        let arr = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let raw = export_to_dlpack(arr).into_raw();
        unsafe { *(*raw).dl_tensor.shape = -2 };
        let result = unsafe { import_from_dlpack::<f64>(raw) };
        assert!(result.is_err());
    }

    #[test]
    fn import_rejects_negative_strides() {
        let arr = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let raw = export_to_dlpack(arr).into_raw();
        unsafe { *(*raw).dl_tensor.strides = -1 };
        let result = unsafe { import_from_dlpack::<f64>(raw) };
        assert!(result.is_err());
    }

    #[test]
    fn import_rejects_misaligned_byte_offset() {
        let arr = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let raw = export_to_dlpack(arr).into_raw();
        unsafe { (*raw).dl_tensor.byte_offset = 4 };
        let result = unsafe { import_from_dlpack::<f64>(raw) };
        assert!(result.is_err());
    }

    #[test]
    fn import_rejects_overflowing_extents() {
        let arr = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let raw = export_to_dlpack(arr).into_raw();
        unsafe {
            *(*raw).dl_tensor.shape = i64::MAX / 4;
            *(*raw).dl_tensor.strides = i64::MAX / 4;
        }
        let result = unsafe { import_from_dlpack::<f64>(raw) };
        assert!(result.is_err());
    }

    #[test]
    fn import_rejects_unknown_device() {
        let arr = NdArray::from_slice(&[1.0, 2.0], &[2]);
        let raw = export_to_dlpack(arr).into_raw();
        // kDLCUDAManaged, a code outside the host-memory set.
        unsafe { (*raw).dl_tensor.device.device_type = DLDeviceType::CUDA_MANAGED };
        let result = unsafe { import_from_dlpack::<f64>(raw) };
        assert!(result.is_err());
    }

    #[test]
    fn import_null_strides_as_row_major() {
        let arr = NdArray::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let raw = export_to_dlpack(arr).into_raw();
        unsafe { (*raw).dl_tensor.strides = std::ptr::null_mut() };
        let imported = unsafe { import_from_dlpack::<f64>(raw) }.unwrap();
        assert_eq!(imported.shape(), &[3, 2]);
        assert_eq!(imported.strides(), &[2, 1]);
        // Row-major reinterpretation of the buffer.
        assert_eq!(imported.get(&[0, 0]), 1.0);
        assert_eq!(imported.get(&[0, 1]), 2.0);
        assert_eq!(imported.get(&[1, 0]), 3.0);
    }

    // *** Foreign producer tensors ************************************

    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Shape and strides storage plus the counter the deleter bumps,
    /// keeping the foreign tensor's pointers alive until release.
    struct ForeignCtx {
        shape: Vec<i64>,
        strides: Vec<i64>,
        hits: &'static AtomicUsize,
    }

    unsafe extern "C" fn counting_deleter(managed: *mut DLManagedTensor) {
        if managed.is_null() {
            return;
        }
        let managed = unsafe { Box::from_raw(managed) };
        let ctx: Box<ForeignCtx> =
            unsafe { Box::from_raw(managed.manager_ctx as *mut ForeignCtx) };
        ctx.hits.fetch_add(1, Ordering::SeqCst);
    }

    /// Assemble a foreign-owned legacy managed tensor over caller-supplied
    /// f64 data, wired to the counting deleter.
    fn foreign_tensor(
        data: *mut c_void,
        shape: Vec<i64>,
        strides: Vec<i64>,
        hits: &'static AtomicUsize,
    ) -> *mut DLManagedTensor {
        let mut ctx = Box::new(ForeignCtx { shape, strides, hits });
        let tensor = DLTensor {
            data,
            device: DLDevice::cpu(),
            ndim: ctx.shape.len() as i32,
            dtype: DLDataType::FLOAT64,
            shape: ctx.shape.as_mut_ptr(),
            strides: ctx.strides.as_mut_ptr(),
            byte_offset: 0,
        };
        Box::into_raw(Box::new(DLManagedTensor {
            dl_tensor: tensor,
            manager_ctx: Box::into_raw(ctx) as *mut c_void,
            deleter: Some(counting_deleter),
        }))
    }

    /// Allocate a 64-byte aligned f64 buffer the test controls outright,
    /// so a zero-copy import shares it without a realignment copy.
    fn aligned_f64_alloc(values: &[f64]) -> (*mut f64, std::alloc::Layout) {
        let layout = std::alloc::Layout::from_size_align(
            values.len() * std::mem::size_of::<f64>(),
            64,
        )
        .unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) as *mut f64 };
        assert!(!ptr.is_null());
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len()) };
        (ptr, layout)
    }

    #[test]
    fn import_rejects_null_shape_pointer() {
        static HITS: AtomicUsize = AtomicUsize::new(0);
        let raw = foreign_tensor(64 as *mut c_void, vec![2], vec![1], &HITS);
        unsafe { (*raw).dl_tensor.shape = std::ptr::null_mut() };
        let result = unsafe { import_from_dlpack::<f64>(raw) };
        let Err(MinarrowError::BridgeError { message, .. }) = result else {
            panic!("expected BridgeError");
        };
        assert!(message.contains("null shape pointer"));
        assert_eq!(HITS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn import_rejects_null_data_pointer() {
        static HITS: AtomicUsize = AtomicUsize::new(0);
        let raw = foreign_tensor(std::ptr::null_mut(), vec![2], vec![1], &HITS);
        let result = unsafe { import_from_dlpack::<f64>(raw) };
        let Err(MinarrowError::BridgeError { message, .. }) = result else {
            panic!("expected BridgeError");
        };
        assert!(message.contains("null data pointer"));
        assert_eq!(HITS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn import_rejects_span_beyond_isize_max() {
        static HITS: AtomicUsize = AtomicUsize::new(0);
        // Validation rejects the span before any read, so the dangling
        // but aligned data pointer is never dereferenced.
        let raw = foreign_tensor(64 as *mut c_void, vec![i64::MAX / 4], vec![1], &HITS);
        let result = unsafe { import_from_dlpack::<f64>(raw) };
        let Err(MinarrowError::BridgeError { message, .. }) = result else {
            panic!("expected BridgeError");
        };
        assert!(message.contains("spans more than isize::MAX bytes"));
        assert_eq!(HITS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn deleter_runs_once_on_drop() {
        static HITS: AtomicUsize = AtomicUsize::new(0);
        let (data, layout) = aligned_f64_alloc(&[1.0, 2.0, 3.0, 4.0]);
        let raw = foreign_tensor(data as *mut c_void, vec![4], vec![1], &HITS);
        let imported = unsafe { import_from_dlpack::<f64>(raw) }.unwrap();
        assert_eq!(imported.get(&[2]), 3.0);
        assert_eq!(HITS.load(Ordering::SeqCst), 0);
        drop(imported);
        assert_eq!(HITS.load(Ordering::SeqCst), 1);
        unsafe { std::alloc::dealloc(data as *mut u8, layout) };
    }

    #[test]
    fn deleter_runs_once_on_failed_import() {
        static HITS: AtomicUsize = AtomicUsize::new(0);
        let (data, layout) = aligned_f64_alloc(&[1.0, 2.0]);
        let raw = foreign_tensor(data as *mut c_void, vec![2], vec![1], &HITS);
        // Importing as f32 against the f64 payload fails the dtype check.
        let result = unsafe { import_from_dlpack::<f32>(raw) };
        assert!(result.is_err());
        assert_eq!(HITS.load(Ordering::SeqCst), 1);
        unsafe { std::alloc::dealloc(data as *mut u8, layout) };
    }

    #[test]
    fn import_mutation_copies_rather_than_writing_through() {
        static HITS: AtomicUsize = AtomicUsize::new(0);
        let (data, layout) = aligned_f64_alloc(&[1.0, 2.0, 3.0, 4.0]);
        let raw = foreign_tensor(data as *mut c_void, vec![4], vec![1], &HITS);
        let mut imported = unsafe { import_from_dlpack::<f64>(raw) }.unwrap();
        imported.set(&[1], 99.0);
        assert_eq!(imported.get(&[1]), 99.0);
        // The producer's memory holds its original values.
        let original = unsafe { std::slice::from_raw_parts(data, 4) };
        assert_eq!(original, &[1.0, 2.0, 3.0, 4.0]);
        drop(imported);
        unsafe { std::alloc::dealloc(data as *mut u8, layout) };
    }

    #[test]
    fn export_aliased_buffer_is_zero_copy() {
        let arr = NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let alias = arr.clone();
        let exported = export_to_dlpack(arr);
        assert_eq!(
            exported.tensor().data as *const f64,
            alias.as_slice().as_ptr()
        );
    }

    #[test]
    fn imported_ndarray_send_drops_on_other_thread() {
        static HITS: AtomicUsize = AtomicUsize::new(0);
        let (data, layout) = aligned_f64_alloc(&[1.0, 2.0, 3.0, 4.0]);
        let raw = foreign_tensor(data as *mut c_void, vec![4], vec![1], &HITS);
        let imported = unsafe { import_from_dlpack::<f64>(raw) }.unwrap();
        std::thread::spawn(move || {
            assert_eq!(imported.get(&[3]), 4.0);
            drop(imported);
        })
        .join()
        .unwrap();
        assert_eq!(HITS.load(Ordering::SeqCst), 1);
        unsafe { std::alloc::dealloc(data as *mut u8, layout) };
    }
}
