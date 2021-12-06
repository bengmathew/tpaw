
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
  throw new Error("Didn't expect to get here")
}

export function nundef<T>(x: T | undefined): T {
  assert(x !== undefined)
  return x as T
}
