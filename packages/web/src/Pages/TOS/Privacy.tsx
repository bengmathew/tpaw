import React, { ReactNode } from 'react'
import { AppPage } from '../App/AppPage'

// Privacy is not part of Contentful because we want finer control over
// timing of changes.
export const Privacy = React.memo(() => {
  return (
    <AppPage
      className=" pt-header min-h-screen"
      title={`Privacy - TPAW Planner`}
    >
      <div className="flex flex-col items-center mb-20 mt-6">
        <div className="w-full max-w-[650px] px-4 z-0">
          <div className=" ">
            {/* <Contentful.RichText
              body={content.TPAW}
              h1="font-bold text-4xl "
              h2="font-bold text-xl mt-6"
              p="mt-4 p-base"
              ul="list-disc ml-5"
            /> */}
            <h1 className="font-bold text-4xl ">Privacy</h1>
            <h2 className="font-bold text-xl mt-6 ">Your Inputs and Results</h2>
            <p className="mt-4 p-base">
              We do not sell any of the data that you enter as inputs or the
              results of the computations.
            </p>
            <p className="mt-4 p-base">
              When you are not logged into an account, your plans are stored on
              the browser and not sent to a server, except in the following
              circumstances:
            </p>
            <ul className="list-disc ml-5">
              <li>
                <p className="mt-4 p-base">
                  When you create a shortened link to a plan, that plan is
                  stored on a server and associated with the link.
                </p>
              </li>
              <li>
                <p className="mt-4 p-base">
                  When you generate a PDF report for a plan, that plan is sent
                  to the server where the report is generated. The plan is used
                  only for generating the report and is not stored on the
                  server. The resulting PDF file is temporarily stored on the
                  server and a download link is provided. This file will be
                  automatically deleted from the server after some time.
                </p>
              </li>
            </ul>
            <p className="mt-4 p-base">
              When you are logged into an account, your plans are stored on a
              server.
            </p>
            <p className="mt-4 p-base">
              Plans stored on the server are saved on{' '}
              <_A href="https://cloud.google.com/">Google Cloud Platform</_A>.
              We use the data associated with these stored plans only as needed
              to implement the functionalities offered by the planner—e.g. to
              load the plan when you login, periodically calculate your updated
              withdrawal amount, etc.{' '}
            </p>
            <p className="mt-4 p-base">
              Your data is always encrypted during transport and storage.
            </p>
            <h2 className="font-bold text-xl mt-6 ">Email</h2>
            <p className="mt-4 p-base">
              When you create an account, you provide us with your email
              address. We use this email address to communicate with you. We do
              not sell your email address to third parties.
            </p>
            <h2 className="font-bold text-xl mt-6 ">Third Parties</h2>
            <p className="mt-4 p-base">
              We use some third parties to provide functionality for the
              website. These third parties may use a variety of tools including
              cookies to track usage and send some data to their servers to
              provide the service. In particular we use:
            </p>
            <ul className="list-disc ml-5">
              <li>
                <p className="mt-4 p-base">
                  <_A href="https://analytics.google.com/">
                    <span className="font-bold">Google Analytics</span>
                  </_A>{' '}
                  to measure traffic and usage of the website. See more on how
                  Google Analytics{' '}
                  <_A href="https://policies.google.com/technologies/partner-sites">
                    uses your data
                  </_A>
                  .
                </p>
              </li>
              <li>
                <p className="mt-4 p-base">
                  <_A href="https://firebase.google.com/support/privacy">
                    <span className="font-bold">
                      Google Firebase Authentication
                    </span>
                  </_A>{' '}
                  for providing authentication services for logging in and
                  signing up users. See more on Firebase{' '}
                  <_A href="https://firebase.google.com/support/privacy">
                    privacy and security
                  </_A>
                  .
                </p>
              </li>
              <li>
                <p className="mt-4 p-base">
                  <_A href="https://sentry.io/">
                    <span className="font-bold">Sentry</span>
                  </_A>{' '}
                  for reporting crashes.
                </p>
              </li>
            </ul>

            <h2 className="font-bold text-xl mt-6 ">Changes</h2>
            <p className="mt-4 p-base">
              We reserve the right to change these policies at any time. If you
              have created an account with us, you will be notified by email
              when a policy changes.
            </p>
          </div>
        </div>
      </div>
    </AppPage>
  )
})

const _A = React.memo(
  ({ children, href }: { children: ReactNode; href: string }) => {
    return (
      <a className="underline" href={href} target="_blank" rel="noreferrer">
        {children}
      </a>
    )
  },
)
