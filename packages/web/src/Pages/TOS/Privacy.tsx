import React from 'react'
import { AppPage } from '../App/AppPage'

export const Privacy = React.memo(() => {
  return (
    <AppPage title="Privacy - TPAW Planner">
      <div className="">
        <h1 className="font-bold text-4xl ">Privacy</h1>
        <h2 className="font-bold text-xl mt-6">{`Your Inputs and Results`}</h2>

        <p className="mt-4">
          {`
          The data that you enter as inputs and the results of the computations
          are not sent to any server. The simulations and other computations are
          performed entirely on your device and do not require communication with
          a server. Any data that is saved is saved exclusively on your device.
          We do not track, store, or sell any of this data.`}
        </p>

        <h2 className="text-lg font-bold mt-6">
          Google Analytics and Crash Reporting
        </h2>
        <p className="mt-4">
          We use Google Analytics{' '}
          (<a href="https://analytics.google.com/" className="underline">
            https://analytics.google.com/
          </a>){' '}
          to measure traffic and usage of the website, and Sentry (
          <a className="underline" href="https://sentry.io">
            https://sentry.io
          </a>
          ) for reporting crashes. These third parties may use a variety of
          tools including cookies to track usage and send some data to their
          servers to provide the service. For more information on how Google
          Analytics uses your data, see{' '}
          <a
            className="underline"
            href="https://policies.google.com/technologies/partner-sites"
          >
            https://policies.google.com/technologies/partner-sites
          </a>
          .
        </p>
      </div>
    </AppPage>
  )
})
