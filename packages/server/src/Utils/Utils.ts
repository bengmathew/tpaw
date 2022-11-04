// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function assert(condition: any): asserts condition {
  if (!condition) {
    throw new Error('Assert')
  }
}

export function assertFalse(): never {
  throw new Error('Assert')
}

export function fGet<T>(x: T | null | undefined): T {
  assert(x !== null && x !== undefined)
  return x as T
}

export function noCase(x: never): never {
  // eslint-disable-next-line @typescript-eslint/restrict-template-expressions, @typescript-eslint/no-explicit-any
  throw new Error(`Didn't expect to get here: ${x as any}`)
}

export function nundef<T>(x: T | undefined): T {
  assert(x !== undefined)
  return x as T
}
