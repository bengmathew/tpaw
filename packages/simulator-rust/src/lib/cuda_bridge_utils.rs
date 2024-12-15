#![allow(non_camel_case_types)]
use crate::cuda_bridge::OptCURRENCY;

// Note: simulator-cuda uses 32 bit floats for non-currency related floating
// point. This narrowing should be done at the cuda boundary and not leak into
// rust. In rust, floats should be f64. This is especially important because 32
// vs 64 bit in cuda is controlled by a flag, so it can be converted to 64 bits
// if needed (eg. testing and benchmarking).

pub type f_cuda_currency = f64;


// EFFICIENT MODE
pub type f_cuda = f32;
pub type f_cuda_currency_npv = f32;

// REPLICATION MODE
// pub type f_cuda = f64;
// pub type f_cuda_currency_npv = f64;

impl From<Option<f_cuda_currency>> for OptCURRENCY {
    fn from(value: Option<f_cuda_currency>) -> Self {
        if let Some(value) = value {
            OptCURRENCY {
                is_set: 1,
                opt_value: value,
            }
        } else {
            OptCURRENCY {
                is_set: 0,
                opt_value: 0.0,
            }
        }
    }
}
