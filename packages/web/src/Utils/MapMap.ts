export const mapMap = <K, V1, V2>(
  map: Map<K, V1>,
  f: (v: V1) => V2,
): Map<K, V2> => new Map<K, V2>([...map.entries()].map(([k, v]) => [k, f(v)]))
