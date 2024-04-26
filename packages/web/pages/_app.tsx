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
import { Spinner } from '../src/Utils/View/Spinner'
import '../styles/globals.css'
import { WithMergeToServer } from '../src/Pages/App/WithMergeToServer'



const MyApp = React.memo(({ Component, pageProps }: AppProps) => {
  const router = useRouter()
  // Hard stop on static site generation. We are happy with static data fetching
  // and client side rendering. This relieves us of having to render a static
  // site in the absence of query parameters and local stores. You need both
  // render and router.isReady to handle case where there are query params, and
  // when there are no query params.
  const [render, setRender] = useState(false)
  useEffect(() => setRender(true), [])
  if (!(render && router.isReady)) return <div></div>

  return (
    <GlobalErrorBoundary>
      <WithFirebaseUser>
        <WithRelayEnvironment>
          <WithMergeToServer>
            <Suspense
              fallback={
                <div className="page h-screen flex flex-col justify-center items-center">
                  <Spinner size="text-4xl" />
                </div>
              }
            >
              <WithSystemInfo>
                <Component {...pageProps} />
              </WithSystemInfo>
            </Suspense>
          </WithMergeToServer>
        </WithRelayEnvironment>
        <ToastContainer position="bottom-center" transition={Slide} />
      </WithFirebaseUser>
    </GlobalErrorBoundary>
  )
})
export default MyApp
