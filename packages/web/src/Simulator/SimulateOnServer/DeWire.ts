import { assert, letIn } from '@tpaw/common'
import _ from 'lodash'

type _Remove_X1xxx<T extends string> = T extends
  | `${infer P}X10`
  | `${infer P}X100`
  | `${infer P}X1000`
  | `${infer P}X10000`
  ? P
  : T
type _Remove_Opt<T extends string> = T extends `${infer P}Opt` ? `${P}` : T
type _Remove_Ms<T extends string> = T extends `${infer P}imestampMs`
  ? `${P}imestamp`
  : T
type _CaseToType<T extends string> = T extends '$case' ? `type` : T

// Maintain same order here as in deWire function.
type _RenameKey<T extends string> = _CaseToType<
  _Remove_Ms<_Remove_X1xxx<_Remove_Opt<T>>>
>

type _ExtractOptKeys<T extends string> = Extract<T, `${string}Opt`>
type _ExtractNonOptKeys<T extends string> = Exclude<T, _ExtractOptKeys<T>>

// On the wire "optional" fields of the protobuf are set to undefined.
export type DeWire<T> = T extends (infer U)[]
  ? DeWire<U>[]
  : T extends object
    ? {
        [P in _ExtractNonOptKeys<string & keyof T> as _RenameKey<
          string & P
        >]: DeWire<T[P]>
      } & {
        [P in _ExtractOptKeys<string & keyof T> as _RenameKey<
          string & P
        >]: DeWire<T[P]> | null
      }
    : Exclude<T, undefined>

type T = DeWire<{
  aX100: number[]
  b: number
}>

export function deWire<T>(
  x: T,
  opts: { scale: number | null; isOpt: boolean } = {
    scale: null,
    isOpt: false,
  },
): DeWire<T> {
  assert(x !== null)
  if (opts.isOpt && x === undefined) {
    return null as DeWire<T>
  }
  assert(x !== undefined)
  if (Array.isArray(x)) {
    return x.map((x) => deWire(x, opts)) as DeWire<T>
  }
  if (typeof x === 'object') {
    return _.fromPairs(
      _.toPairs(x).map(([key, v]) => {
        const newOpts = {
          ...opts,
          scale: null as number | null,
          isOpt: false,
        }
        const transforms: ((key: string) => string)[] = [
          (key) => {
            const match = key.match(/^(.+)Opt$/)
            if (match) {
              newOpts.isOpt = true
              return `${match[1]}`
            } else {
              return key
            }
          },
          (key) => {
            const match = key.match(/^(.+)X1(0+)$/)
            if (match) {
              newOpts.scale = 1 / Math.pow(10, match[2].length)
              return match[1]
            } else {
              return key
            }
          },
          (key: string) =>
            letIn({ match: key.match(/^(.+)imestampMs$/) }, ({ match }) =>
              match ? `${match[1]}imestamp` : key,
            ),
          (key: string) => (key === '$case' ? 'type' : key),
        ]
        const modifiedKey = transforms.reduce(
          (key, transform) => transform(key),
          key,
        )

        try {
          const deWired = deWire(v, newOpts)
          return [modifiedKey, deWired]
        } catch (e) {
          console.log('deWire error', { key, v, newOpts, x })
          throw e
        }
      }),
    ) as DeWire<T>
  }
  if (typeof x === 'number') {
    return (opts.scale === null ? x : x * opts.scale) as DeWire<T>
  }
  return x as DeWire<T>
}
