import _ from 'lodash'
import { assert } from './Utils'

type MapType<Type> = {
  [Property in keyof Type]: Type[Property] extends (x: unknown) => infer U
    ? U
    : never
}

export type Validator<T, F = unknown> = (x: F) => T
export namespace Validator {
  export class Failed extends Error {
    path
    lines
    constructor(lines: string | string[], path?: string) {
      super('')
      this.lines = _.flatten([lines])
      this.path = path
    }
    get fullLines() {
      return this.path
        ? [`Property ${this.path}:`, ...this.lines.map((x) => `   ${x}`)]
        : this.lines
    }
    get fullMessage() {
      return this.fullLines.join('\n')
    }
  }

  export const number =
    () =>
    (x: unknown): number => {
      if (typeof x !== 'number') throw new Failed('Not a number.')
      return x
    }
  export const boolean =
    () =>
    (x: unknown): boolean => {
      if (typeof x !== 'boolean') throw new Failed('Not a boolean.')
      return x
    }
  export const string =
    () =>
    (x: unknown): string => {
      if (typeof x !== 'string') throw new Failed('Not a string.')
      return x
    }
  export const constant =
    <C extends number | string | null | boolean>(c: C) =>
    (x: unknown): C => {
      if (x !== c) {
        // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
        const message = `Not ${c === null ? 'null' : `"${c}"`}.`
        throw new Failed(message)
      }
      return c
    }

  export const array =
    <T>(test: Validator<T>) =>
    (x: unknown): T[] => {
      if (!_.isArray(x)) throw new Failed('Not an array.')
      const result = [] as T[]
      return x.map((e, i) => {
        try {
          return test(e)
        } catch (e) {
          assert(e !== null)
          if (e instanceof Failed) {
            throw new Failed([
              `At index ${i}:`,
              ...e.fullLines.map((x) => `    ${x}`),
            ])
          } else {
            throw e
          }
        }
      })
    }

  export const object =
    <O extends Record<string, Validator<any>>>(tests: O) =>
    (x: unknown): MapType<O> => {
      if (!_.isObject(x) || _.isArray(x) || _.isFunction(x))
        throw new Failed('Not an object.')
      const anyX = x as any
      const missingKeys = _.difference(_.keys(tests), _.keys(x))
      if (missingKeys.length > 0) {
        throw new Failed(
          `Missing ${
            missingKeys.length === 1 ? 'property' : 'properties'
          } ${missingKeys.join(', ')}.`,
        )
      }
      const result = _.mapValues(tests, (test, key) => {
        try {
          return test(anyX[key])
        } catch (e) {
          assert(e !== null)
          if (e instanceof Failed) {
            throw new Failed(e.lines, `${key}${e.path ? `.${e.path}` : ''}`)
          } else {
            throw e
          }
        }
      })

      return result
    }

  export const union =
    <T extends Validator<any>[]>(...tests: T): Validator<MapType<T>[number]> =>
    (x: unknown) => {
      const messages = [] as string[]
      let i = 0
      for (const test of tests) {
        try {
          return test(x)
        } catch (e) {
          assert(e !== null)
          if (e instanceof Failed) {
            messages.push(
              `Option ${i + 1}:`,
              ...e.fullLines.map((x) => `    ${x}`),
            )
          } else {
            throw e
          }
        }
        i++
      }
      throw new Failed(messages)
    }

  export function intersection<T1, T2>(
    ...tests: [Validator<T1>, Validator<T2>]
  ): Validator<T1 & T2>
  export function intersection<T1, T2, T3>(
    ...tests: [Validator<T1>, Validator<T2>, Validator<T3>]
  ): Validator<T1 & T2 & T3>
  export function intersection(...tests: any[]) {
    return (x: any) => {
      let result: any = {}
      for (const test of tests) {
        result = { ...result, ...test(x) }
      }
      return result
    }
  }

  export function chain<T0, T1, T2>(
    ...tests: [Validator<T1, T0>, Validator<T2, T1>]
  ): Validator<T2, T0>
  export function chain<T0, T1, T2, T3>(
    ...tests: [Validator<T1, T0>, Validator<T2, T1>, Validator<T3, T2>]
  ): Validator<T3, T0>
  export function chain<T0, T1, T2, T3>(
    ...tests: [Validator<T1, T0>, Validator<T2, T1>, Validator<T3, T2>]
  ): Validator<T3, T0>
  export function chain<T0, T1, T2, T3, T4>(
    ...tests: [
      Validator<T1, T0>,
      Validator<T2, T1>,
      Validator<T3, T2>,
      Validator<T4, T3>,
    ]
  ): Validator<T4, T0>
  export function chain(...tests: any[]) {
    return (x: any) => {
      let result: any = x
      for (const test of tests) {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
        result = test(result)
      }
      return result
    }
  }
}
