import { faSpider } from '@fortawesome/pro-light-svg-icons'
import { faCopy, faEnvelope } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import * as Sentry from '@sentry/nextjs'
import { block, noCase } from '@tpaw/common'
import { useRouter } from 'next/dist/client/router'
import Head from 'next/head'
import Image from 'next/image'
import Link from 'next/link'
import React, { ReactNode, useCallback, useEffect, useState } from 'react'
import { ToastOptions } from 'react-toastify'
import { createContext } from '../../Utils/CreateContext'
import { errorToast } from '../../Utils/CustomToasts'
import { ErrorBoundary } from '../../Utils/ErrorBoundary'
import { useAssertConst } from '../../Utils/UseAssertConst'
import { Config } from '../Config'
import { AppError } from './AppError'
import { Footer } from './Footer'
import clsx from 'clsx'

const [Context, useSetGlobalError] = createContext<{
  setGlobalError: (error: Error) => void
}>('GlobalErrorCallback')
export { useSetGlobalError }

export const useDefaultErrorHandlerForNetworkCall = () => {
  const { setGlobalError } = useSetGlobalError()
  const defaultErrorHandlerForNetworkCall = useCallback(
    ({
      e,
      toast,
      toastId,
    }: {
      e: Error
      toast: string | null
      toastId?: string
    }) => {
      Sentry.captureException(e)
      if (e instanceof AppError) {
        const toastOpts: ToastOptions = { toastId }
        if (e.code === 'serverDownForMaintenance') {
          errorToast(
            'Could not complete action. The server is down for maintenace.',
            toastOpts,
          )
          return
        } else if (e.code === 'serverDownForUpdate') {
          errorToast(
            'Could not complete action. The server is updating to a new version.',
            toastOpts,
          )
          return
        } else if (e.code === 'clientNeedsUpdate') {
          setGlobalError(e)
          return
        }
      }
      if (toast) errorToast(toast)
    },
    [setGlobalError],
  )
  useAssertConst([setGlobalError])
  return {
    defaultErrorHandlerForNetworkCall,
  }
}

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
  !Config.client.isProduction && console.dir(error)

  const [errorId, setErrorId] = useState<string | null>(null)
  const router = useRouter()
  useEffect(() => {
    setErrorId(Sentry.captureException(error))
    router.events.on('routeChangeComplete', () => window.location.reload())
  }, [error, router])

  useEffect(() => {
    ;(window as any).__APP_ERROR__ = true
  }, [])

  return (
    <div className="min-h-screen page">
      <Head>
        <title>Error - TPAW Planner</title>
      </Head>
      <div className="grid h-screen" style={{ grid: '1fr auto/auto' }}>
        <div className="flex flex-col justify-center items-center p-4 mb-10">
          <div className="max-w-[500px]">
            {error instanceof AppError ? (
              error.code === 'concurrentChange' ? (
                <_Custom
                  title="Concurrent Change"
                  message="It looks like this plan is being updated concurrently on
                another device or browser tab. Please reload to sync
                changes."
                  action="reload"
                />
              ) : error.code === '404' ? (
                <_Custom title="404" message="Page not found." action="home" />
              ) : error.code === 'serverDownForMaintenance' ? (
                <_Custom
                  title="Down for Maintenance"
                  message="The server is down for maintenace. We will be back online
                shortly."
                  action="reload"
                />
              ) : error.code === 'serverDownForUpdate' ? (
                <_Custom
                  title=" Updating to a New Version"
                  message="We are updating to new version of the planner. We will be
                  back online shortly."
                  action="reload"
                />
              ) : error.code === 'clientNeedsUpdate' ? (
                <_Custom
                  title=" New Version Available"
                  message="A new version of the planner is now available. Please reload to get the lastest version."
                  action="reload"
                />
              ) : error.code === 'networkError' ? (
                <_Custom
                  title=" Network Error"
                  message="Cloud not access the network. Please check your internet connection."
                  action="reload"
                />
              ) : error.code === '413' || error.code === 'serverError' ? (
                <_SomethingWentWrong errorId={errorId} />
              ) : (
                noCase(error.code)
              )
            ) : (
              <_SomethingWentWrong errorId={errorId} />
            )}
          </div>
        </div>
        <Footer />
      </div>
    </div>
  )
})

const _SomethingWentWrong = React.memo(
  ({ className, errorId }: { className?: string; errorId: string | null }) => {
    const emailHref = block(() => {
      // Using URL object was not formatting correctly.
      const subject = `TPAWPlanner Crashed`
      const body = `Hi Ben%2C%0A%0A${
        errorId ? `Error: ${errorId}%0A%0A` : ''
      }I had an issue while using the planner:%0A%0A`
      return `mailto:ben@tpawplanner.com?subject=${subject}&body=${body}`
    })
    return (
      <div className={clsx(className)}>
        <div className="flex items-center gap-x-4">
          <FontAwesomeIcon className="text-[40px] " icon={faSpider} />
          <p className="font-font2  text-xl">Something went wrong</p>
        </div>
        <div className="mt-4 border-t-2  border-gray-700 ">
          <p className="p-base mt-4">
            We would love to know what happened so we can try to fix the issue.
          </p>
          <p className="p-base mt-4">
            We are usually able to fix all issues that we can reproduce, so the
            most helpful information we can get is the sequence of steps that
            generates the crash. But even if you are not able to identify the
            sequence of steps, we would still love to hear from you — we might
            still be able to fix the problem.
          </p>
          {errorId && (
            <>
              <p className="p-base mt-4">
                You can reference the following error ID to help us identify the
                corresponding crash report.
              </p>
              {/* <p className="font-font2 text-base lighten mt-4 ">
                Error: {errorId}
              </p> */}
              <button
                className="py-1 px-2 rounded-md border border-gray-400 mt-4 text-sm flex items-center lighten"
                onClick={() => {
                  void navigator.clipboard.writeText(errorId)
                }}
              >
                Error: {errorId}
                <FontAwesomeIcon className="ml-2 text-base" icon={faCopy} />
              </button>
            </>
          )}
          <p className="p-base mt-4">You can contact us at:</p>
          <div className="mt-2 ">
            {/* <h2 className="font-bold text-2xl">Contact</h2> */}

            <a
              className="flex items-center py-2 cursor-pointer"
              href={emailHref}
              target="_blank"
              rel="noreferrer"
              onClick={() => {}}
            >
              <span className="inline-block w-[25px]">
                <FontAwesomeIcon icon={faEnvelope} />
              </span>
              <span className="underline">ben@tpawplanner.com</span>
            </a>
            <a
              className="flex items-center py-2 cursor-pointer"
              href="https://www.bogleheads.org/forum/viewtopic.php?t=331368"
            >
              <div className="w-[25px]">
                <Image
                  src="/bolgeheads_logo.png"
                  alt="substack icon"
                  width="17"
                  height="15"
                />
              </div>
              <h2 className="underline">Bogleheads</h2>
            </a>
          </div>
        </div>
      </div>
    )
  },
)

const _Custom = React.memo(
  ({
    title,
    message,
    action,
  }: {
    title: string
    message: string
    action: 'reload' | 'home'
  }) => {
    return (
      <>
        <h2 className="font-bold text-xl">{title}</h2>
        <p className="p-base text-xl mt-6">{message}</p>
        {action === 'reload' ? (
          <button
            className="btn-dark btn2-md mt-6"
            onClick={() => window.location.reload()}
          >
            Reload
          </button>
        ) : action === 'home' ? (
          <Link
            className="inline-block btn-dark btn2-md mt-6"
            href={Config.client.urls.app()}
          >
            Home
          </Link>
        ) : (
          noCase(action)
        )}
      </>
    )
  },
)
