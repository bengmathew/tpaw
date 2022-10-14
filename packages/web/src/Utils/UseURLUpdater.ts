import {useRouter} from 'next/router'
import {useState} from 'react'

// This is to get around router.push not being const.
export function useURLUpdater() {
  const router = useRouter()
  const [result] = useState(() => ({
    push: (url: URL) => void router.push(url, url, {shallow: true}),
    replace: (url: URL) => void router.replace(url, url, {shallow: true}),
  }))
  return result
}
