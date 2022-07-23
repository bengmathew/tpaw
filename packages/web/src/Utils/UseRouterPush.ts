import {useRouter} from 'next/router'
import {useState} from 'react'

export function useURLUpdater() {
  const router = useRouter()

  const [result] = useState(() => ({
    push: (url: URL) => void router.push(url),
  }))
  return result
}
