import type {AppProps} from 'next/app'
import {useRouter} from 'next/dist/client/router'
import React, {useEffect, useState} from 'react'
import {ToastContainer} from 'react-toastify'
import 'react-toastify/dist/ReactToastify.css'
import {GlobalErrorBoundary} from '../src/Pages/App/GlobalErrorBoundary'
import {WithChartData} from '../src/Pages/App/WithChartData'
import {WithSimulation} from '../src/Pages/App/WithSimulation'
import '../styles/globals.css'

/**
 * String.prototype.replaceAll() polyfill
 * https://gomakethings.com/how-to-replace-a-section-of-a-string-with-another-one-with-vanilla-js/
 * @author Chris Ferdinandi
 * @license MIT
 */
if (!String.prototype.replaceAll) {
  ;(String as any).prototype.replaceAll = function (
    str: string | RegExp,
    newStr: string
  ) {
    // If a regex pattern
    if (
      Object.prototype.toString.call(str).toLowerCase() === '[object regexp]'
    ) {
      return this.replace(str, newStr)
    }

    // If a string
    return this.replace(new RegExp(str, 'g'), newStr)
  }
}

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
        <WithChartData>
          <Component {...pageProps} />
        </WithChartData>
      </WithSimulation>

      <ToastContainer />
    </GlobalErrorBoundary>
  )
}
export default MyApp
