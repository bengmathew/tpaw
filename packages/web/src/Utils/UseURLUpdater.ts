import { useRouter } from 'next/router'
import { useState } from 'react'

// This is to get around router not being const.
export function useURLUpdater() {
  const router = useRouter()
  const [result] = useState(() => ({
    push: (url: URL | string) => void router.push(url, url, { shallow: true }),
    replace: (url: URL | string) =>
      void router.replace(url, url, { shallow: true }),
  }))
  return result
}
