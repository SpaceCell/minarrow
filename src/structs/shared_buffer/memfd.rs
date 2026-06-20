// Copyright 2025-2026 Peter Garfield Bower. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # **MemfdBuffer** - *Zero-copy cross-process buffer sharing via memfd*
//!
//! Linux-only buffer backed by `memfd_create()` for sharing memory between processes.
//!
//! ## Purpose
//! When you need to share Arrow data between processes without copying:
//! 1. Parent creates Table with memfd-backed buffers
//! 2. Parent passes file descriptors to child processes
//! 3. Children mmap the same fds - same physical memory, zero copy
//!
//! ## Usage
//! ```rust,ignore
//! // Parent process
//! let memfd = MemfdBuffer::new("my_buffer", 1024)?;
//! let fd = memfd.fd();  // Pass this to child process
//!
//! // Use with SharedBuffer
//! let shared = SharedBuffer::from_owner(memfd);
//!
//! // Child process (after receiving fd)
//! let memfd = MemfdBuffer::reopen(parent_pid, fd, 1024)?;
//! let shared = SharedBuffer::from_owner(memfd);
//! // Same physical memory, zero copy!
//! ```
//!
//! ## Alignment
//! Buffers are guaranteed 64-byte aligned for SIMD operations.
//! Extra space is allocated and an offset computed to achieve alignment.
//!
//! ## Platform
//! Linux only - `memfd_create()` is a Linux-specific syscall.

use std::io;
use std::os::unix::io::{AsRawFd, RawFd};

/// Buffer backed by a memfd for zero-copy cross-process sharing.
///
/// The underlying memory is created via `memfd_create()` and memory-mapped.
/// The file descriptor can be shared with other processes, which can then
/// mmap the same physical memory for true zero-copy access.
///
/// # Alignment
/// The buffer is guaranteed to be 64-byte aligned for SIMD operations.
/// This is achieved by allocating extra space and computing an offset.
///
/// # Lifecycle
/// - The memfd is created when `new()` is called
/// - The fd remains open as long as the MemfdBuffer exists
/// - When dropped, the mmap is unmapped and fd is closed
/// - If other processes have mmap'd the fd, their mappings remain valid
///   (kernel keeps the memfd alive until all references are gone)
pub struct MemfdBuffer {
    /// Raw pointer to the aligned data region
    ptr: *mut u8,
    /// Original (page-aligned) pointer returned by `mmap`. Required for
    /// `munmap` because `ptr` is offset to a 64-byte boundary.
    mmap_ptr: *mut u8,
    /// Usable length in bytes
    len: usize,
    /// Total mmap'd size (includes alignment padding)
    mmap_len: usize,
    /// File descriptor for the memfd (for sharing with other processes)
    fd: RawFd,
}

// SAFETY: The memfd memory is anonymous and can be safely shared between threads.
// The ptr is valid for the lifetime of the MemfdBuffer.
unsafe impl Send for MemfdBuffer {}
unsafe impl Sync for MemfdBuffer {}

impl MemfdBuffer {
    /// Create a new memfd-backed buffer with the specified size.
    ///
    /// The buffer is guaranteed to be 64-byte aligned for SIMD operations.
    ///
    /// # Arguments
    /// * `name` - Name for the memfd - visible in `/proc/pid/fd/`
    /// * `size` - Requested size in bytes
    ///
    /// # Returns
    /// A new MemfdBuffer or IO error
    ///
    /// # Platform
    /// Linux only
    #[cfg(target_os = "linux")]
    pub fn new(name: &str, size: usize) -> io::Result<Self> {
        use std::ffi::CString;

        if size == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Size must be greater than 0",
            ));
        }

        // Allocate extra space for alignment (worst case: 63 bytes of padding)
        let total_size = size + 64;

        let c_name = CString::new(name).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("Invalid name: {}", e))
        })?;

        // Create memfd with sealing enabled so we can lock the size below.
        let fd = unsafe { libc::memfd_create(c_name.as_ptr(), libc::MFD_ALLOW_SEALING) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }

        // Set size
        if unsafe { libc::ftruncate(fd, total_size as libc::off_t) } < 0 {
            let err = io::Error::last_os_error();
            unsafe { libc::close(fd) };
            return Err(err);
        }

        // Seal the fd against grow/shrink. Consumers (including ourselves)
        // can still read and write the contents but can no longer change the
        // backing length, so any mapping we hand out remains valid for the
        // life of the buffer.
        let seals = libc::F_SEAL_SHRINK | libc::F_SEAL_GROW;
        if unsafe { libc::fcntl(fd, libc::F_ADD_SEALS, seals) } < 0 {
            let err = io::Error::last_os_error();
            unsafe { libc::close(fd) };
            return Err(err);
        }

        // mmap the memfd
        let mmap_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };

        if mmap_ptr == libc::MAP_FAILED {
            let err = io::Error::last_os_error();
            unsafe { libc::close(fd) };
            return Err(err);
        }

        // Calculate alignment offset to achieve 64-byte alignment
        let base_addr = mmap_ptr as usize;
        let alignment_offset = (64 - (base_addr % 64)) % 64;
        let aligned_ptr = unsafe { (mmap_ptr as *mut u8).add(alignment_offset) };

        Ok(Self {
            ptr: aligned_ptr,
            mmap_ptr: mmap_ptr as *mut u8,
            len: size,
            mmap_len: total_size,
            fd,
        })
    }

    /// Reopen an existing memfd from another process.
    ///
    /// Uses `/proc/{pid}/fd/{fd}` to access the memfd from the creator process.
    ///
    /// # Arguments
    /// * `creator_pid` - PID of the process that created the memfd
    /// * `fd` - File descriptor number in the creator's process
    /// * `size` - Expected size of the usable buffer region
    ///
    /// # Returns
    /// A MemfdBuffer backed by the same memory, or IO error
    ///
    /// # Security
    /// Requires either:
    /// - Same user as the creator process, or
    /// - CAP_SYS_PTRACE capability
    ///
    /// # Platform
    /// Linux only
    #[cfg(target_os = "linux")]
    pub fn reopen(creator_pid: u32, fd: RawFd, size: usize) -> io::Result<Self> {
        use std::fs::OpenOptions;

        let path = format!("/proc/{}/fd/{}", creator_pid, fd);

        let file = OpenOptions::new().read(true).write(true).open(&path)?;

        // Match what `new()` allocates: caller's requested size plus the
        // 64-byte alignment padding window. The producer-declared length is
        // verified to be at least this much, but we never map more than the
        // alignment-padded window we asked for - a peer's outsized memfd
        // can't be used to balloon our address space.
        let needed = size
            .checked_add(64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "size + 64 overflows"))?;
        let actual_size = file.metadata()?.len() as usize;

        if actual_size < needed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Memfd too small: expected at least {} bytes, got {}",
                    needed, actual_size
                ),
            ));
        }
        let total_size = needed;

        // mmap the file
        let our_fd = file.as_raw_fd();
        let mmap_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                our_fd,
                0,
            )
        };

        if mmap_ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        // Keep the fd open (don't let File drop close it)
        let owned_fd = unsafe {
            // Duplicate the fd so we own it
            let dup_fd = libc::dup(our_fd);
            if dup_fd < 0 {
                libc::munmap(mmap_ptr, total_size);
                return Err(io::Error::last_os_error());
            }
            dup_fd
        };

        // Calculate alignment offset (must match what new() computed)
        let base_addr = mmap_ptr as usize;
        let alignment_offset = (64 - (base_addr % 64)) % 64;
        let aligned_ptr = unsafe { (mmap_ptr as *mut u8).add(alignment_offset) };

        Ok(Self {
            ptr: aligned_ptr,
            mmap_ptr: mmap_ptr as *mut u8,
            len: size,
            mmap_len: total_size,
            fd: owned_fd,
        })
    }

    /// Get the raw file descriptor for sharing with other processes.
    ///
    /// The fd can be passed to child processes which can then call
    /// `MemfdBuffer::reopen()` to access the same memory.
    #[inline]
    pub fn fd(&self) -> RawFd {
        self.fd
    }

    /// Get the usable length in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the data as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Get the data as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl AsRef<[u8]> for MemfdBuffer {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsMut<[u8]> for MemfdBuffer {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl Drop for MemfdBuffer {
    fn drop(&mut self) {
        // Unmap from the original mmap base. `self.ptr` is offset to a 64-byte
        // boundary inside this allocation, so deriving the base from it would
        // be lossy if `mmap` ever returns a non-64-aligned address.
        unsafe {
            libc::munmap(self.mmap_ptr as *mut libc::c_void, self.mmap_len);
        }

        // Close the fd
        unsafe {
            libc::close(self.fd);
        }
    }
}

//
// Memfd-specific vtable for SharedBuffer
//

use crate::Vec64;
use crate::structs::shared_buffer::SharedBuffer;
use crate::structs::shared_buffer::internal::vtable::Vtable;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

/// Reference-counted wrapper for MemfdBuffer.
/// This is identical to Owned<T> but we know the concrete type.
#[repr(C)]
pub(crate) struct OwnedMemfd {
    pub(crate) ref_cnt: AtomicUsize,
    pub(crate) owner: MemfdBuffer,
}

/// Clones memfd buffer by incrementing reference count.
unsafe fn memfd_clone(h: &AtomicPtr<()>, p: *const u8, l: usize) -> SharedBuffer {
    let raw = h.load(Ordering::Acquire);
    assert!(!raw.is_null());
    let ref_cnt = unsafe { &*(raw as *const AtomicUsize) };
    ref_cnt.fetch_add(1, Ordering::Relaxed);
    SharedBuffer {
        ptr: p,
        len: l,
        data: AtomicPtr::new(raw),
        vtable: &MEMFD_VT,
    }
}

/// Decrements reference count, deallocating if last reference.
unsafe fn memfd_drop(h: &mut AtomicPtr<()>, _p: *const u8, _l: usize) {
    let raw = h.load(Ordering::Acquire);
    if raw.is_null() {
        return;
    }
    let ref_cnt = unsafe { &*(raw as *const AtomicUsize) };
    if ref_cnt.fetch_sub(1, Ordering::AcqRel) == 1 {
        drop(unsafe { Box::from_raw(raw as *mut OwnedMemfd) });
    }
}

/// Checks if this is the only reference.
unsafe fn memfd_is_unique(h: &AtomicPtr<()>) -> bool {
    let raw = h.load(Ordering::Acquire);
    if raw.is_null() {
        return false;
    }
    let ref_cnt = unsafe { &*(raw as *const AtomicUsize) };
    ref_cnt.load(Ordering::Acquire) == 1
}

/// Extracts the memfd file descriptor from the owned buffer.
unsafe fn memfd_get_fd(h: &AtomicPtr<()>) -> Option<i32> {
    let raw = h.load(Ordering::Acquire);
    if raw.is_null() {
        return None;
    }
    let owned = unsafe { &*(raw as *const OwnedMemfd) };
    Some(owned.owner.fd)
}

/// Vtable for memfd-backed buffers with fd extraction support.
pub(crate) static MEMFD_VT: Vtable = Vtable {
    clone: memfd_clone,
    drop: memfd_drop,
    is_unique: memfd_is_unique,
    to_vec: |_, p, l| unsafe { std::slice::from_raw_parts(p, l) }.to_vec(),
    to_vec64: |_, p, l| {
        let mut v = Vec64::with_capacity(l);
        unsafe {
            v.extend_from_slice(std::slice::from_raw_parts(p, l));
        }
        v
    },
    memfd_fd: memfd_get_fd,
};

impl SharedBuffer {
    /// Constructs a `SharedBuffer` from a `MemfdBuffer` with fd extraction support.
    ///
    /// Unlike `from_owner()`, this preserves the ability to extract the memfd
    /// file descriptor via `memfd_fd()`.
    pub fn from_memfd_owner(memfd: MemfdBuffer) -> Self {
        let raw: *mut OwnedMemfd = Box::into_raw(Box::new(OwnedMemfd {
            ref_cnt: AtomicUsize::new(1),
            owner: memfd,
        }));
        let buf = unsafe { (*raw).owner.as_ref() };
        Self {
            ptr: buf.as_ptr(),
            len: buf.len(),
            data: AtomicPtr::new(raw.cast()),
            vtable: &MEMFD_VT,
        }
    }

    /// Returns the memfd file descriptor if this buffer is backed by a memfd.
    ///
    /// Returns `None` if the buffer is not memfd-backed.
    #[inline]
    pub fn memfd_fd(&self) -> Option<i32> {
        unsafe { (self.vtable.memfd_fd)(&self.data) }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn test_create_memfd() {
        let memfd = MemfdBuffer::new("test_buffer", 1024).expect("Failed to create memfd");
        assert_eq!(memfd.len(), 1024);
        assert!(!memfd.is_empty());
        assert!(memfd.fd() >= 0);
    }

    #[test]
    fn test_alignment() {
        let memfd = MemfdBuffer::new("align_test", 1024).expect("Failed to create memfd");
        let ptr_addr = memfd.as_slice().as_ptr() as usize;
        assert_eq!(ptr_addr % 64, 0, "Buffer should be 64-byte aligned");
    }

    #[test]
    fn test_read_write() {
        let mut memfd = MemfdBuffer::new("rw_test", 256).expect("Failed to create memfd");

        // Write some data
        let data = b"Hello, memfd!";
        memfd.as_mut_slice()[..data.len()].copy_from_slice(data);

        // Read it back
        assert_eq!(&memfd.as_slice()[..data.len()], data);
    }

    #[test]
    fn test_reopen_same_process() {
        let memfd = MemfdBuffer::new("reopen_test", 512).expect("Failed to create memfd");
        let fd = memfd.fd();
        let pid = std::process::id();

        // Write some data
        let data = b"Test data for reopen";
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), memfd.ptr, data.len());
        }

        // Reopen from the same process (simulates child process access)
        let reopened = MemfdBuffer::reopen(pid, fd, 512).expect("Failed to reopen memfd");

        // Verify we see the same data
        assert_eq!(&reopened.as_slice()[..data.len()], data);
    }

    #[test]
    fn test_zero_size_rejected() {
        let result = MemfdBuffer::new("zero_test", 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_with_shared_buffer() {
        let memfd = MemfdBuffer::new("shared_test", 256).expect("Failed to create memfd");

        // This should work because MemfdBuffer implements AsRef<[u8]>
        let shared = SharedBuffer::from_owner(memfd);
        assert_eq!(shared.len(), 256);
    }

    #[test]
    fn test_memfd_fd_extraction() {
        let memfd = MemfdBuffer::new("fd_test", 256).expect("Failed to create memfd");
        let expected_fd = memfd.fd();

        // Use from_memfd_owner to preserve fd extraction
        let shared = SharedBuffer::from_memfd_owner(memfd);
        assert_eq!(shared.len(), 256);

        // Should be able to extract the fd
        let extracted_fd = shared.memfd_fd();
        assert_eq!(extracted_fd, Some(expected_fd));
    }

    #[test]
    fn test_memfd_fd_extraction_from_buffer() {
        use crate::structs::buffer::Buffer;

        let buffer: Buffer<i64> =
            Buffer::from_memfd("buffer_fd_test", 100).expect("Failed to create memfd buffer");

        // Should be able to extract the fd
        let fd = buffer.memfd_fd();
        assert!(fd.is_some());
        assert!(fd.unwrap() >= 0);
    }

    #[test]
    fn test_cloned_buffer_preserves_fd() {
        let memfd = MemfdBuffer::new("clone_test", 256).expect("Failed to create memfd");
        let expected_fd = memfd.fd();

        let shared = SharedBuffer::from_memfd_owner(memfd);
        let cloned = shared.clone();

        // Both should have the same fd
        assert_eq!(shared.memfd_fd(), Some(expected_fd));
        assert_eq!(cloned.memfd_fd(), Some(expected_fd));
    }

    #[test]
    fn test_non_memfd_buffer_returns_none() {
        // Regular SharedBuffer (not memfd-backed) should return None
        let shared = SharedBuffer::from_vec(vec![1, 2, 3, 4]);
        assert_eq!(shared.memfd_fd(), None);
    }
}
