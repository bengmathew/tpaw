import { useRouter } from 'next/router'
import { useMemo } from 'react'

export const useURLWithParams = (param: string, value: string) => {
  const path = useRouter().asPath
  return useMemo(() => {
    const url = new URL(path, window.location.origin)
    url.searchParams.set(param, value)
    return url
  }, [param, path, value])
}

export const useURLPathWithoutParams = (param: string) => {
  const path = useRouter().asPath
  return useMemo(() => {
    const url = new URL(path, window.location.origin)
    url.searchParams.delete(param)
    return url
  }, [param, path])
}
