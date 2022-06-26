import {useRouter} from 'next/router'
import {useMemo} from 'react'

export function useURLUpdater() {
  const router = useRouter()

  const result = useMemo(
    () => ({
      push: (url: URL) => void router.push(url),
    }),
    
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  )
  return result
}
