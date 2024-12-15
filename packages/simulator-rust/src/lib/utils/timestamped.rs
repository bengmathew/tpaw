pub trait Timestamped {
    fn timestamp_ms(&self) -> i64;
}

pub fn get_data_for_timestamp<T: Timestamped>(series: &[T], timestamp_ms: i64) -> &T {
    series
        .iter()
        .rev()
        .find(|x| timestamp_ms >= x.timestamp_ms())
        .unwrap()
}
