import { faSpider } from '@fortawesome/pro-thin-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import * as Sentry from '@sentry/nextjs'
import { noCase } from '@tpaw/common'
import { useRouter } from 'next/dist/client/router'
import Head from 'next/head'
import React, { ReactNode, useEffect, useState } from 'react'
import { createContext } from '../../Utils/CreateContext'
import { ErrorBoundary } from '../../Utils/ErrorBoundary'
import { Config } from '../Config'
import { AppError } from './AppError'
import { Footer } from './Footer'

type Value = { setGlobalError: (error: Error) => void }
const [Context, useSetGlobalError] = createContext<Value>('GlobalErrorCallback')
export { useSetGlobalError }

export const GlobalErrorBoundary = React.memo(
  ({ children }: { children: ReactNode }) => {
    const [error, setGlobalError] = useState<Error | null>(null)
    useEffect(() => {
      window.onunhandledrejection = (e) => {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
        setGlobalError(e.reason)
      }
    }, [])
    if (error) return <_ErrorFallback error={error} />
    return (
      <Context.Provider value={{ setGlobalError }}>
        <ErrorBoundary fallback={(error) => <_ErrorFallback error={error} />}>
          {children}
        </ErrorBoundary>
      </Context.Provider>
    )
  },
)

export const _ErrorFallback = React.memo(({ error }: { error: Error }) => {
  !Config.client.production && console.dir(error)

  const router = useRouter()
  useEffect(() => {
    Sentry.captureException(error)
    router.events.on('routeChangeComplete', () => window.location.reload())
  }, [error, router])

  return (
    <div className="min-h-screen page">
      <Head>
        <title>Error - TPAW Planner</title>
      </Head>
      <div className="grid h-screen" style={{ grid: '1fr auto/auto' }}>
        <div className="flex flex-col items-center mb-10">
          <div className="flex h-full flex-col justify-center items-center p-4">
            <div className="flex items-center gap-x-4">
              <FontAwesomeIcon className="text-[60px]" icon={faSpider} />
              <h2 className="text-xl ">{_message(error)}</h2>
            </div>

            {error instanceof AppError && error.code === 'invalidParameters' ? (
              <div className="w-full">
                <h2 className="text-lg mt-6">You can reset inputs to start over. </h2>
                <button
                  className="btn-dark btn-sm mt-6"
                  onClick={() => {
                    window.localStorage.removeItem('params')
                    window.location.reload()
                  }}
                >
                  Reset All Inputs
                </button>
              </div>
            ) : (
              <button
                className="btn-dark btn-sm mt-6"
                onClick={() => window.location.reload()}
              >
                Reload
              </button>
            )}
          </div>
        </div>
        <Footer />
      </div>
    </div>
  )
})

const _message = (e: Error): string => {
  if (e instanceof AppError) {
    switch (e.code) {
      case 'invalidParameters':
        return 'Something went wrong with the inputs.'
      default:
        noCase(e.code)
    }
  } else {
    return 'Something went wrong.'
  }
}
