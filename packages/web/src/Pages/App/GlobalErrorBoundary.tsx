import { faSpider } from '@fortawesome/pro-thin-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import * as Sentry from '@sentry/nextjs'
import { assertFalse, noCase } from '@tpaw/common'
import { useRouter } from 'next/dist/client/router'
import Head from 'next/head'
import React, { ReactNode, useEffect, useState } from 'react'
import { createContext } from '../../Utils/CreateContext'
import { ErrorBoundary } from '../../Utils/ErrorBoundary'
import { Config } from '../Config'
import { AppError } from './AppError'
import { Footer } from './Footer'

const [Context, useSetGlobalError] = createContext<{
  setGlobalError: (error: Error) => void
}>('GlobalErrorCallback')
export { useSetGlobalError }

// Hack does not work if it is an Error object.
export const errorToKillNavigation_ignore = 'errorToKillNavigation_ignore'

export const GlobalErrorBoundary = React.memo(
  ({ children }: { children: ReactNode }) => {
    const [error, setGlobalError] = useState<Error | null>(null)
    useEffect(() => {
      window.onunhandledrejection = (e) => {
        // FROM: https://github.com/vercel/next.js/discussions/32231#discussioncomment-6351802
        if (e.reason === errorToKillNavigation_ignore) {
          e.preventDefault()
        } else {
          // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
          setGlobalError(e.reason)
        }
      }
      window.onerror = (event, source, lineno, colno, error) => {
        setGlobalError(
          error instanceof Error
            ? error
            : // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
              new Error(`Error type was not error: ${error}`),
        )
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
        <div className="flex flex-col justify-center items-center p-4 mb-10">
          <div className="max-w-[500px]">
            {error instanceof AppError && error.code === 'concurrentChange' ? (
              <>
                <h2 className="font-bold text-xl">Concurrent Change</h2>
                <p className="p-base text-xl mt-6">
                  It looks like this plan is being updated concurrently on
                  another device or browser tab. Please reload to sync changes.
                </p>
                <button
                  className="btn-dark btn2-md mt-6"
                  onClick={() => window.location.reload()}
                >
                  Reload
                </button>
              </>
            ) : (
              <>
                <div className="flex items-center gap-x-4">
                  <FontAwesomeIcon className="text-[50px]" icon={faSpider} />
                  <p className="font-font2 text-xl">{_message(error)}</p>
                </div>

                {error instanceof AppError &&
                error.code === 'concurrentChange' ? (
                  <div className="w-full">
                    <h2 className="text-lg mt-6">
                      You can reset inputs to start over.{' '}
                    </h2>
                    <button
                      className="btn-dark btn-sm mt-6"
                      onClick={() => {
                        window.localStorage.removeItem('params')
                        window.location.href = `${Config.client.urls.app(
                          '/path',
                        )}`
                        window.location.reload()
                      }}
                    >
                      Reset All Inputs
                    </button>
                  </div>
                ) : (
                  <button
                    className="btn-dark btn2-md mt-6"
                    onClick={() => window.location.reload()}
                  >
                    Reload
                  </button>
                )}
              </>
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
      case 'concurrentChange':
        assertFalse()
      case 'serverDownForMaintenance':
        return 'The server is down for maintenace. We will be back online shortly.'
      case 'networkError':
        return 'There was a problem connecting with the server. Please reload to try again.'
      case '404':
        return '404. Page not found.'
      default:
        noCase(e.code)
    }
  } else {
    return 'Something went wrong.'
  }
}
