import {useRouter} from 'next/router'
import {fGet} from './Utils'

export function useURLParam(key: string): string | null {
  const result = useRouter().query[key]
  if (result instanceof Array) throw new Error()
  return result ?? null
}

export const useFURLParam = (key: string) => fGet(useURLParam(key))
