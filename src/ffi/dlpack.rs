//! # **DLPack FFI** - *Zero-copy tensor interchange with PyTorch, TensorFlow, JAX, and others*
//!
//! Implements the [DLPack](https://github.com/dmlc/dlpack) tensor interchange
//! standard for NdArray, enabling zero-copy data sharing across language and
//! framework boundaries.
//!
//! ## Supported protocols
//! - **DLManagedTensor** (v0.x) - capsule-based ownership transfer
//! - Export: [`export_to_dlpack`] wraps an NdArray as a DLManagedTensor
//! - Import: [`import_from_dlpack`] wraps a foreign DLManagedTensor as an NdArray
//!
//! ## Layout compatibility
//! NdArray's column-major layout with 64-byte aligned strides is expressed
//! through DLPack's strides field. Column-major is fully supported by
//! PyTorch and NumPy with zero copy. TensorFlow may copy to row-major
//! internally.
//!
//! ## Notes
//! - DLPack uses element strides, matching NdArray's convention
//! - DLPack has no null mask concept - NaN-based missing values align with this
//! - The 64-byte padding between columns is transparent to consumers via strides

use std::ffi::c_void;
use std::sync::Arc;

use crate::structs::buffer::Buffer;
use crate::structs::ndarray::NdArray;
use crate::structs::shared_buffer::SharedBuffer;

// ****************************************************************
// DLPack C structs (matching the DLPack header)
// ****************************************************************

/// Device type codes from the DLPack spec.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DLDeviceType {
    CPU = 1,
    CUDA = 2,
    CUDAHost = 3,
    ROCm = 10,
    ROCmHost = 11,
    Metal = 8,
    Vulkan = 7,
    OpenCL = 4,
}

/// Device descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDevice {
    pub device_type: DLDeviceType,
    pub device_id: i32,
}

/// Data type descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DLDataType {
    /// Type code: 0=int, 1=uint, 2=float, 4=bfloat, 5=complex, 6=bool
    pub code: u8,
    /// Number of bits per element
    pub bits: u8,
    /// Number of SIMD lanes, typically 1
    pub lanes: u16,
}

impl DLDataType {
    /// f64 type descriptor.
    pub const FLOAT64: Self = DLDataType { code: 2, bits: 64, lanes: 1 };
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

/// DLPack v0.x managed tensor with ownership semantics.
/// The deleter is called when the consumer is done with the tensor.
#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

unsafe impl Send for DLManagedTensor {}
unsafe impl Sync for DLManagedTensor {}

// ****************************************************************
// Export: NdArray -> DLManagedTensor
// ****************************************************************

/// Keeps the NdArray and the i64 shape/strides arrays alive for the
/// lifetime of the exported DLManagedTensor.
struct DLPackHolder {
    _ndarray: Arc<NdArray>,
    shape_i64: Vec<i64>,
    strides_i64: Vec<i64>,
}

/// Export an NdArray as a DLPack managed tensor for zero-copy sharing.
///
/// Returns an `DLPackTensor` that manages the lifecycle. The NdArray's
/// buffer stays alive via Arc reference counting. When the owned tensor is
/// dropped, the backing data is released.
///
/// For FFI handoff (e.g. creating a PyCapsule), call `.into_raw()` to
/// transfer ownership to the consumer.
pub fn export_to_dlpack(ndarray: Arc<NdArray>) -> DLPackTensor {
    let ndim = ndarray.ndim();
    let shape = ndarray.shape();
    let strides = ndarray.strides();

    let shape_i64: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
    let strides_i64: Vec<i64> = strides.iter().map(|&s| s as i64).collect();

    let data_ptr = ndarray.as_slice().as_ptr() as *mut c_void;

    let mut holder = Box::new(DLPackHolder {
        _ndarray: ndarray,
        shape_i64,
        strides_i64,
    });

    let tensor = DLTensor {
        data: data_ptr,
        device: DLDevice { device_type: DLDeviceType::CPU, device_id: 0 },
        ndim: ndim as i32,
        dtype: DLDataType::FLOAT64,
        shape: holder.shape_i64.as_mut_ptr(),
        strides: holder.strides_i64.as_mut_ptr(),
        byte_offset: 0,
    };

    let managed = Box::new(DLManagedTensor {
        dl_tensor: tensor,
        manager_ctx: Box::into_raw(holder) as *mut c_void,
        deleter: Some(dlpack_deleter),
    });

    DLPackTensor { ptr: Box::into_raw(managed) }
}

/// Release callback invoked by the foreign consumer when it is done
/// with the tensor. Drops the holder which releases the Arc<NdArray>.
///
/// # Safety
/// Must only be called once per DLManagedTensor.
unsafe extern "C" fn dlpack_deleter(managed: *mut DLManagedTensor) {
    if managed.is_null() { return; }
    let managed = unsafe { Box::from_raw(managed) };
    if !managed.manager_ctx.is_null() {
        let _holder: Box<DLPackHolder> = unsafe {
            Box::from_raw(managed.manager_ctx as *mut DLPackHolder)
        };
        // holder drops here, releasing the Arc<NdArray>
    }
}

// ****************************************************************
// DLPackTensor - safe wrapper with Drop
// ****************************************************************

/// Safe wrapper owning a DLPack managed tensor. Calls the deleter on drop.
///
/// This is the return type of `NdArray::to_dlpack()`, analogous to how
/// `to_apache_arrow()` returns an `ArrayRef` that owns its lifecycle.
pub struct DLPackTensor {
    ptr: *mut DLManagedTensor,
}

impl DLPackTensor {
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

// ****************************************************************
// Import: DLManagedTensor -> NdArray
// ****************************************************************

/// Wrap that owns the foreign DLManagedTensor and calls its deleter on drop.
///
/// The DLPack contract makes the deleter responsible for releasing the
/// managed tensor allocation, so the raw pointer is held rather than a
/// `Box`. Wrapping it in a `Box` would free the allocation a second time
/// after the deleter already released it.
struct ForeignDLPack {
    ptr: *const u8,
    len_bytes: usize,
    managed: *mut DLManagedTensor,
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
        if self.managed.is_null() {
            return;
        }
        unsafe {
            if let Some(deleter) = (*self.managed).deleter {
                deleter(self.managed);
            }
        }
        self.managed = std::ptr::null_mut();
    }
}

unsafe impl Send for ForeignDLPack {}
unsafe impl Sync for ForeignDLPack {}

/// Import a foreign DLManagedTensor as a Minarrow NdArray.
///
/// Only CPU f64 tensors are supported. The NdArray takes ownership of the
/// DLManagedTensor and will call its deleter when dropped.
///
/// # Safety
/// - The DLManagedTensor must be valid and point to accessible CPU memory
/// - The caller transfers ownership - the tensor must not be used after this call
/// - The tensor must contain f64 data
pub unsafe fn import_from_dlpack(managed_ptr: *mut DLManagedTensor) -> Result<NdArray, String> {
    if managed_ptr.is_null() {
        return Err("DLPack: null pointer".to_string());
    }

    // Take ownership of the managed tensor up front. Any early return drops
    // this guard, which calls the foreign deleter and releases the tensor as
    // the ownership-transfer contract requires. On the success path the guard
    // moves into the SharedBuffer instead.
    let mut foreign = ForeignDLPack {
        ptr: std::ptr::null(),
        len_bytes: 0,
        managed: managed_ptr,
    };

    let tensor = unsafe { &(*managed_ptr).dl_tensor };

    // Validate device
    if tensor.device.device_type != DLDeviceType::CPU {
        return Err(format!(
            "DLPack: only CPU tensors supported, got device type {:?}",
            tensor.device.device_type
        ));
    }

    // Validate dtype
    if tensor.dtype.code != 2 || tensor.dtype.bits != 64 || tensor.dtype.lanes != 1 {
        return Err(format!(
            "DLPack: only f64 supported, got code={} bits={} lanes={}",
            tensor.dtype.code, tensor.dtype.bits, tensor.dtype.lanes
        ));
    }

    let ndim = tensor.ndim as usize;
    if ndim == 0 {
        return Err("DLPack: 0-dimensional tensors not supported".to_string());
    }

    // Read shape
    let shape: Vec<usize> = unsafe {
        std::slice::from_raw_parts(tensor.shape, ndim)
            .iter()
            .map(|&s| s as usize)
            .collect()
    };

    // Read strides (null means C-contiguous i.e. row-major)
    let strides: Vec<usize> = if tensor.strides.is_null() {
        // Row-major strides
        let mut s = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            s[d] = s[d + 1] * shape[d + 1];
        }
        s
    } else {
        unsafe {
            std::slice::from_raw_parts(tensor.strides, ndim)
                .iter()
                .map(|&s| s as usize)
                .collect()
        }
    };

    // Compute total buffer length needed
    let max_offset: usize = shape.iter()
        .zip(strides.iter())
        .map(|(&s, &st)| if s == 0 { 0 } else { (s - 1) * st })
        .sum();
    let buf_len = if shape.iter().any(|&s| s == 0) { 0 } else { max_offset + 1 };

    // Data pointer with byte offset
    let data_ptr = unsafe {
        (tensor.data as *const u8).add(tensor.byte_offset as usize)
    } as *const f64;

    let buf_len_bytes = buf_len * std::mem::size_of::<f64>();

    // Record the data window now that shape and strides are known.
    foreign.ptr = data_ptr as *const u8;
    foreign.len_bytes = buf_len_bytes;

    // Wrap into SharedBuffer -> Buffer -> NdArray. The guard moves into the
    // SharedBuffer, so the deleter now runs when the NdArray is dropped.
    let shared = SharedBuffer::from_owner(foreign);
    let buffer: Buffer<f64> = Buffer::from_shared(shared);

    Ok(NdArray::from_buffer(buffer, &shape, &strides))
}

// ****************************************************************
// Tests
// ****************************************************************

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_roundtrip_1d() {
        let arr = Arc::new(NdArray::from_slice(&[1.0, 2.0, 3.0], &[3]));
        let owned = export_to_dlpack(arr.clone());
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
        let arr = Arc::new(NdArray::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]
        ));
        let owned = export_to_dlpack(arr.clone());
        let tensor = owned.tensor();

        assert_eq!(tensor.ndim, 2);

        let shape = unsafe { std::slice::from_raw_parts(tensor.shape, 2) };
        assert_eq!(shape[0], 3);
        assert_eq!(shape[1], 2);

        let strides = unsafe { std::slice::from_raw_parts(tensor.strides, 2) };
        assert_eq!(strides[0], 1);
        assert_eq!(strides[1], 8);

        let data = tensor.data as *const f64;
        assert_eq!(unsafe { *data }, 1.0);
        assert_eq!(unsafe { *data.add(2) }, 3.0);
        assert_eq!(unsafe { *data.add(strides[1] as usize) }, 4.0);
        assert_eq!(unsafe { *data.add(strides[1] as usize + 2) }, 6.0);
    }

    #[test]
    fn export_preserves_data_lifetime() {
        let owned = {
            let arr = Arc::new(NdArray::from_slice(&[42.0, 99.0], &[2]));
            export_to_dlpack(arr)
        };
        let data = owned.tensor().data as *const f64;
        assert_eq!(unsafe { *data }, 42.0);
        assert_eq!(unsafe { *data.add(1) }, 99.0);
    }

    #[test]
    fn import_from_exported() {
        let original = Arc::new(NdArray::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]
        ));
        let owned = export_to_dlpack(original.clone());
        let raw = owned.into_raw();

        let imported = unsafe { import_from_dlpack(raw) }.unwrap();
        assert_eq!(imported.shape(), &[3, 2]);
        assert_eq!(imported.get(&[0, 0]), 1.0);
        assert_eq!(imported.get(&[2, 0]), 3.0);
        assert_eq!(imported.get(&[0, 1]), 4.0);
        assert_eq!(imported.get(&[2, 1]), 6.0);
    }

    #[test]
    fn import_null_ptr_fails() {
        let result = unsafe { import_from_dlpack(std::ptr::null_mut()) };
        assert!(result.is_err());
    }

    #[test]
    fn export_3d_strides() {
        let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let arr = Arc::new(NdArray::from_slice(&data, &[2, 3, 4]));
        let owned = export_to_dlpack(arr);
        let tensor = owned.tensor();

        assert_eq!(tensor.ndim, 3);

        let shape = unsafe { std::slice::from_raw_parts(tensor.shape, 3) };
        assert_eq!(shape, &[2, 3, 4]);

        let strides = unsafe { std::slice::from_raw_parts(tensor.strides, 3) };
        assert_eq!(strides[0], 1);
        assert_eq!(strides[1], 8);
        assert_eq!(strides[2], 24);
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
}
