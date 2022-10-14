import _ from 'lodash'
import {useRouter} from 'next/router'
import {fGet} from './Utils'

export function useURLParam(key: string): string | null {
  const result = useRouter().query[key]
  return _.first(result instanceof Array ? result : [result]) ?? null
}

export const useFURLParam = (key: string) => fGet(useURLParam(key))
