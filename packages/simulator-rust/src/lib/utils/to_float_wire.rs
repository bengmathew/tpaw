// EFFICIENT MODE
type FloatWire = i64;
#[inline(always)]
fn round_f64(x: f64) -> f64 {
    x.round()
}
#[inline(always)]
fn round_f32(x: f32) -> f32 {
    x.round()
}

// REPLICATION MODE
// type FloatWire = f64;
// #[inline(always)]
// fn round_f64(x: f64) -> f64 {
//     x
// }
// #[inline(always)]
// fn round_f32(x: f32) -> f32 {
//     x
// }


// ---------- ToFloatWire ----------
pub trait ToFloatWire {
    fn to_float_wire(&self, x: i64) -> FloatWire;
}
impl ToFloatWire for f64 {
    #[inline(always)]
    fn to_float_wire(&self, x: i64) -> FloatWire {
        (round_f64(*self * x as f64)) as FloatWire
    }
}
impl ToFloatWire for f32 {
    #[inline(always)]
    fn to_float_wire(&self, x: i64) -> FloatWire {
        (round_f32(*self * x as f32)) as FloatWire
    }
}


// ---------- ToFloatWireVec ----------
pub trait ToFloatWireVec {
    fn to_float_wire(&self, x: i64) -> Vec<FloatWire>;
}
impl ToFloatWireVec for Vec<f64> {
    #[inline(always)]
    fn to_float_wire(&self, x: i64) -> Vec<FloatWire> {
        self.iter().map(|y| y.to_float_wire(x)).collect()
    }
}
impl ToFloatWireVec for Vec<f32> {
    #[inline(always)]
    fn to_float_wire(&self, x: i64) -> Vec<FloatWire> {
        self.iter().map(|y| y.to_float_wire(x)).collect()
    }
}
