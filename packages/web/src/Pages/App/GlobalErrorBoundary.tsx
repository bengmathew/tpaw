import { faSpider } from '@fortawesome/pro-light-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
// import * as Sentry from '@sentry/nextjs'
import { useRouter } from 'next/dist/client/router'
import React, { ReactNode, useEffect, useState } from 'react'
import { createContext } from '../../Utils/CreateContext'
import { ErrorBoundary } from '../../Utils/ErrorBoundary'
import { Config } from '../Config'
import { AppError } from './AppError'
import { AppPage } from './AppPage'

type Value = {setGlobalError: (error: Error) => void}
const [Context, useSetGlobalError] = createContext<Value>('GlobalErrorCallback')
export { useSetGlobalError }

export const GlobalErrorBoundary = React.memo(
  ({children}: {children: ReactNode}) => {
    const [error, setGlobalError] = useState<Error | null>(null)
    if (error) return <_ErrorFallback error={error} />
    return (
      <Context.Provider value={{setGlobalError}}>
        <ErrorBoundary fallback={error => <_ErrorFallback error={error} />}>
          {children}
        </ErrorBoundary>
      </Context.Provider>
    )
  }
)

export const _ErrorFallback = React.memo(({error}: {error: Error}) => {
  !Config.client.production && console.dir(error)

  const router = useRouter()
  useEffect(() => {
    // Not using sentry for now because it does not yet work on Next.js v12.
    // Sentry.captureException(error)
    router.events.on('routeChangeComplete', () => window.location.reload())
  }, [error, router])


  return (
    <AppPage title="TPAW Planner - Error">
      <div className="flex h-full flex-col justify-center items-center p-4">
        <div className="flex items-center gap-x-4">
          <FontAwesomeIcon className="text-[60px]" icon={faSpider} />
          <h2 className="text-xl ">{_message(error)}</h2>
        </div>
      </div>
    </AppPage>
  )
})

const _message = (e: Error) => {
  if (e instanceof AppError) {
    return e.message
  } else {
    return 'Something went wrong.'
  }
}
