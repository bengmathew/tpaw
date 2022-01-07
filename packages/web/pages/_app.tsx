import type {AppProps} from 'next/app'
import {useRouter} from 'next/dist/client/router'
import React, {useEffect, useState} from 'react'
import {GlobalErrorBoundary} from '../src/Pages/App/GlobalErrorBoundary'
import {WithSimulation} from '../src/Pages/App/WithSimulation'
import '../styles/globals.css'

function MyApp({Component, pageProps}: AppProps) {
  const router = useRouter()
  // Hard stop on SSG because suspense is not (yet) supported. You need both
  // render and router.isReady to handle case where there are query params,
  // and when there are no query params.
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
