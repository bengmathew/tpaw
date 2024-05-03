import { PlanPaths } from '@tpaw/common'
import _ from 'lodash'
import { useRouter } from 'next/router'
import { useCallback, useEffect, useRef, useState } from 'react'
import { errorToKillNavigation_ignore } from '../Pages/App/GlobalErrorBoundary'

// Based on: https://github.com/vercel/next.js/discussions/32231#discussioncomment-6351802
export const useNavGuard = (isSyncing: boolean, planPaths: PlanPaths) => {
  const router = useRouter()
  const [state, setState] = useState<
    | { isTriggered: false }
    | { isTriggered: true; isBrowserNav: true }
    | { isTriggered: true; isBrowserNav: false; timestamp: number; url: URL }
  >({ isTriggered: false })

  const handleSyncing = () => {
    // This does not get displayed as far as I can tell. This is just a signal
    // to some browsers to actually show the alert that ends up using browser
    // default text.
    const warningText = 'Unsaved changes. Are you sure you want to leave?'
    const windowHandler = (e: BeforeUnloadEvent) => {
      setState({ isTriggered: true, isBrowserNav: true })
      e.preventDefault()
      return (e.returnValue = warningText)
    }
    window.addEventListener('beforeunload', windowHandler)

    const routerHandler = (path: string) => {
      const parts = planPaths().pathname.split('/')
      if (
        _.isEqual(parts, path.split('?')[0].split('/').slice(0, parts.length))
      )
        return
      setState({
        isTriggered: true,
        isBrowserNav: false,
        timestamp: Date.now(),
        url: new URL(path, window.location.origin),
      })
      router.events.emit('routeChangeError')
      throw errorToKillNavigation_ignore
    }
    router.events.on('routeChangeStart', routerHandler)

    return () => {
      window.removeEventListener('beforeunload', windowHandler)
      router.events.off('routeChangeStart', routerHandler)
    }
  }
  const handleSyncingRef = useRef(handleSyncing)
  handleSyncingRef.current = handleSyncing

  const resetNavGuardState = useCallback(
    () => setState({ isTriggered: false }),
    [],
  )
  useEffect(() => {
    if (!isSyncing) return
    return handleSyncingRef.current()
  }, [isSyncing])
  return {
    navGuardState: state,
    resetNavGuardState,
  }
}
