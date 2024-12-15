pub trait FGetItemAtOrBeforeKey<T> {
    fn fget_item_at_or_before_key<K, F>(&self, key: K, f: F) -> &T
    where
        F: Fn(&T) -> K,
        K: Ord;
}

impl<T> FGetItemAtOrBeforeKey<T> for Vec<T> {
    fn fget_item_at_or_before_key<K, F>(&self, key: K, f: F) -> &T
    where
        F: Fn(&T) -> K,
        K: Ord,
    {
        match self.binary_search_by_key(&key, f) {
            Ok(index) => &self[index],
            Err(index) => self.get(index - 1).unwrap(),
        }
    }
}

impl<T> FGetItemAtOrBeforeKey<T> for [T] {
    fn fget_item_at_or_before_key<K, F>(&self, key: K, f: F) -> &T
    where
        F: Fn(&T) -> K,
        K: Ord,
    {
        match self.binary_search_by_key(&key, f) {
            Ok(index) => &self[index],
            Err(index) => self.get(index - 1).unwrap(),
        }
    }
}
