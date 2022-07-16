import {faSpider} from '@fortawesome/pro-light-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import * as Sentry from '@sentry/nextjs'
import {useRouter} from 'next/dist/client/router'
import React, {ReactNode, useEffect, useState} from 'react'
import {createContext} from '../../Utils/CreateContext'
import {ErrorBoundary} from '../../Utils/ErrorBoundary'
import {Config} from '../Config'
import {AppError} from './AppError'
import {AppPage} from './AppPage'
import {Footer} from './Footer'

type Value = {setGlobalError: (error: Error) => void}
const [Context, useSetGlobalError] = createContext<Value>('GlobalErrorCallback')
export {useSetGlobalError}

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
    Sentry.captureException(error)
    router.events.on('routeChangeComplete', () => window.location.reload())
  }, [error, router])

  return (
    <AppPage className="min-h-screen" title={'Error - TPAW Planner'} curr={'other'}>
      <div className="grid h-screen" style={{grid: '1fr auto/auto'}}>
        <div className="flex flex-col items-center mb-10">
          <div className="flex h-full flex-col justify-center items-center p-4">
            <div className="flex items-center gap-x-4">
              <FontAwesomeIcon className="text-[60px]" icon={faSpider} />
              <h2 className="text-xl ">{_message(error)}</h2>
            </div>
            <h2 className="">
              If this keeps happening try:{' '}
              <button className="btn-dark btn-sm" onClick={() => {
                window.localStorage.removeItem('params')
                window.location.reload()
              }}>
                Reset All Inputs
              </button>
            </h2>
          </div>
        </div>
        <Footer />
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
