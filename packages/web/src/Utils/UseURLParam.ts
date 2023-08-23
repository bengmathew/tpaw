import _ from 'lodash'
import { useRouter } from 'next/router'
import { assert, fGet } from './Utils'

export function useURLParam(key: string): string | null {
  const result = useRouter().query[key]
  if (result instanceof Array) {
    assert(result.length === 1)
  }
  return _.first(result instanceof Array ? result : [result]) ?? null
}

export const useFURLParam = (key: string) => fGet(useURLParam(key))
