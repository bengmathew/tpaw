import '../src/Utils/StringReplaceAllPolyfill'

import type { AppProps } from 'next/app'
import { useRouter } from 'next/dist/client/router'
import React, { Suspense, useEffect, useState } from 'react'
import { Slide, ToastContainer } from 'react-toastify'
import 'react-toastify/dist/ReactToastify.css'
import { GlobalErrorBoundary } from '../src/Pages/App/GlobalErrorBoundary'
import { WithFirebaseUser } from '../src/Pages/App/WithFirebaseUser'
import { WithRelayEnvironment } from '../src/Pages/App/WithRelayEnvironment'
import { WithSystemInfo } from '../src/Pages/App/WithSystemInfo'
import '../styles/globals.css'
import { WithMergeToServer } from '../src/Pages/App/WithMergeToServer'
import { createContext } from '../src/Utils/CreateContext'

import { Spinner } from '../src/Utils/View/Spinner'
import { clsx } from 'clsx'

// This is a hack to get around not being able to suspend in WithSimulation
// until the first simulation is completed. Alternative is to render the
// suspense component inside WithSimulation, but that means the spinner resets
// because it is a different component.
const [GlobalSuspenseFallbackContext, useGlobalSuspenseFallbackContext] =
  createContext<{
    setGlobalSuspend: (suspend: boolean) => void
  }>('GlobalSuspenseFallbackContext')
export { useGlobalSuspenseFallbackContext }

const MyApp = React.memo(({ Component, pageProps }: AppProps) => {
  const router = useRouter()
  // Hard stop on static site generation. We are happy with static data fetching
  // and client side rendering. This relieves us of having to render a static
  // site in the absence of query parameters and local stores. You need both
  // render and router.isReady to handle case where there are query params, and
  // when there are no query params.
  const [render, setRender] = useState(false)
  useEffect(() => setRender(true), [])

  const [globalSuspend, setGlobalSuspend] = useState(false)
  if (!(render && router.isReady)) return <div></div>

  return (
    <GlobalErrorBoundary>
      <WithFirebaseUser>
        <WithRelayEnvironment>
          <WithMergeToServer>
            <GlobalSuspenseFallbackContext.Provider
              value={{ setGlobalSuspend }}
            >
              <Suspense
                fallback={
                  <TriggerSuspenseFallback
                    setGlobalSuspend={setGlobalSuspend}
                  />
                }
              >
                <WithSystemInfo>
                  <Component {...pageProps} />
                </WithSystemInfo>
              </Suspense>
            </GlobalSuspenseFallbackContext.Provider>
            {/* After the component, so it will be on top. */}
            <_TopLevelSuspenseFallback show={globalSuspend} />
          </WithMergeToServer>
        </WithRelayEnvironment>
        <ToastContainer position="bottom-center" transition={Slide} />
      </WithFirebaseUser>
    </GlobalErrorBoundary>
  )
})
export default MyApp

const TriggerSuspenseFallback = ({
  setGlobalSuspend,
}: {
  setGlobalSuspend: (suspend: boolean) => void
}) => {
  useEffect(() => {
    setGlobalSuspend(true)
    return () => setGlobalSuspend(false)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  return <></>
}

const _TopLevelSuspenseFallback = ({ show }: { show: boolean }) => {
  return (
    <div
      className={clsx(
        'page absolute top-0 left-0 w-full h-screen flex flex-col justify-center items-center bg-white',
        show ? 'block' : 'hidden',
      )}
    >
      <Spinner size="text-4xl" />
    </div>
  )
}
