//! `Tensor<'a, T>`: a typed, shaped view into a `Region`.
//!
//! `device_ptr()` is the only way to reach a raw pointer, and it borrows
//! `self`. Two tensors cannot hold `&mut` into the same region without
//! the borrow checker catching it.

use core::marker::PhantomData;

use rvllm_core::{DType, Shape};

use crate::graph_safe::GraphSafe;
use crate::hbm::Region;

/// Typed view into a region. `T` is the element type; `dtype` is
/// carried separately so FP8 views can use a `u8`-sized element with a
/// distinct dtype tag.
pub struct Tensor<'a, T> {
    region: &'a Region<'a>,
    shape: Shape,
    dtype: DType,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T> Tensor<'a, T> {
    /// Build a tensor view over a region. Caller is responsible that
    /// `shape.numel() * dtype.bytes() <= region.len()` — checked here,
    /// with overflow guards. Without checked_mul a 4-D shape like
    /// `[u32::MAX, 2, 2, 2]` would silently wrap in release and let
    /// the assertion accept a 0-byte region as "fits".
    pub fn new(region: &'a Region<'a>, shape: Shape, dtype: DType) -> Self {
        let numel = shape
            .numel_checked()
            .expect("Tensor::new shape numel overflow");
        let needed = numel
            .checked_mul(dtype.bytes())
            .expect("Tensor::new byte-size overflow");
        assert!(
            needed <= region.len(),
            "Tensor: shape {:?} @ {:?} needs {} B but region '{}' is {} B",
            shape,
            dtype,
            needed,
            region.name(),
            region.len(),
        );
        Self {
            region,
            shape,
            dtype,
            _phantom: PhantomData,
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Device pointer of element `[0,0,...,0]`. Borrows `self` so the
    /// pointer cannot outlive the view.
    pub fn device_ptr(&self) -> u64 {
        self.region.device_ptr()
    }

    /// Element byte stride along each axis (row-major).
    pub fn byte_strides(&self) -> [usize; rvllm_core::MAX_RANK] {
        let mut s = self.shape.strides();
        let b = self.dtype.bytes();
        for x in s.iter_mut() {
            *x *= b;
        }
        s
    }
}

// `Tensor<'a, T>` is GraphSafe iff its region is, which we've already
// guaranteed. Capture may bind `&Tensor`.
unsafe impl<'a, T> GraphSafe for Tensor<'a, T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hbm::HbmArena;

    #[test]
    fn tensor_ptr_matches_region_ptr() {
        let arena = HbmArena::new_host_stub(1 << 20);
        let r = arena.region("t", 4096, 16).unwrap();
        let t: Tensor<half::f16> = Tensor::new(&r, Shape::new(&[8, 128]), DType::F16);
        assert_eq!(t.device_ptr(), r.device_ptr());
        assert_eq!(t.byte_strides()[0], 128 * 2);
        assert_eq!(t.byte_strides()[1], 2);
    }

    #[test]
    #[should_panic(expected = "needs")]
    fn oversized_shape_panics() {
        let arena = HbmArena::new_host_stub(4096);
        let r = arena.region("small", 16, 1).unwrap();
        // 16 f16 = 32 B > region 16 B
        let _: Tensor<half::f16> = Tensor::new(&r, Shape::new(&[16]), DType::F16);
    }
}
