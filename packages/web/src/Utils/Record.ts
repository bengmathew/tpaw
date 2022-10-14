import _ from 'lodash'
export namespace Record {


  export const fromPairs = <Key extends string, Value>(
    x: readonly (readonly [Key, Value])[]
  ) =>
    _.fromPairs(x as unknown as readonly [Key, Value][]) as Record<Key, Value>

  export const toPairs = <Key extends string, Value>(x: Record<Key, Value>) =>
    _.toPairs(x) as [Key, Value][]

  export const mapValues = <Key extends string, Value1, Value2>(
    x: Record<Key, Value1>,
    fn: (v: Value1, k: Key) => Value2
  ): Record<Key, Value2> => fromPairs(toPairs(x).map(([k, v]) => [k, fn(v, k)]))

  export const map = <Key1 extends string, Key2 extends string, Value1, Value2>(
    x: Record<Key1, Value1>,
    fn: (k: Key1, v: Value1) => readonly [Key2, Value2]
  ): Record<Key2, Value2> => fromPairs(toPairs(x).map(([k, v]) => fn(k, v)))

  export const merge = <Key1 extends string, Key2 extends string, Value>(
    x: Record<Key1, Value>,
    y: Record<Key2, Value>
  ) => ({...x, ...y} as Record<Key1 | Key2, Value>)
}
