import type {AppProps} from 'next/app'
import {useRouter} from 'next/dist/client/router'
import React, {useEffect, useState} from 'react'
import {GlobalErrorBoundary} from '../src/Pages/App/GlobalErrorBoundary'
import {WithSimulation} from '../src/Pages/App/WithSimulation'
import '../styles/globals.css'

function MyApp({Component, pageProps}: AppProps) {
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
      <WithSimulation>
        <Component {...pageProps} />
      </WithSimulation>
    </GlobalErrorBoundary>
  )
}
export default MyApp
