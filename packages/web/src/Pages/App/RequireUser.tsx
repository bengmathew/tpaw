import { faGoogle } from '@fortawesome/free-brands-svg-icons'
import { faEnvelope } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { API } from '@tpaw/common'
import { getAuth, GoogleAuthProvider, signInWithRedirect } from 'firebase/auth'
import Head from 'next/head'
import Link from 'next/link'
import React, { Dispatch, ReactNode, useMemo, useRef, useState } from 'react'
import { graphql, useMutation } from 'react-relay'
import { fGet, noCase } from '../../Utils/Utils'
import { Spinner } from '../../Utils/View/Spinner'
import { RequireUserSendEmailMutation } from './__generated__/RequireUserSendEmailMutation.graphql'
import { useDefaultErrorHandlerForNetworkCall } from './GlobalErrorBoundary'
import { useFirebaseUser } from './WithFirebaseUser'

export const RequireUser = React.memo(
  ({ children }: { children: ReactNode }) => {
    const firebaseUser = useFirebaseUser()
    const user = useMemo(
      () => (firebaseUser ? { firebaseUser: firebaseUser } : null),
      [firebaseUser],
    )
    if (!user) return <_Login />
    return <>{children}</>
  },
)

type _State = { type: 'input' } | { type: 'emailSent' }
const _Login = React.memo(() => {
  const [state, setState] = useState<_State>({ type: 'input' })
  const [email, setEmail] = useState('')
  return (
    <div
      className={`page grid justify-center items-center h-screen overflow-scroll`}
      style={{ grid: 'auto/auto' }}
    >
      <Head>
        <title>Login - TPAW Planner</title>
      </Head>
      <div className="py-10 w-screen max-w-[400px] px-4">
        {state.type === 'input' ? (
          <_LoginInput
            className=""
            email={email}
            setEmail={setEmail}
            onEmailSent={() => {
              setState({ type: 'emailSent' })
              window.localStorage.setItem('emailForSignIn', email)
            }}
          />
        ) : state.type === 'emailSent' ? (
          <_LoginEmailSent className="px-4" email={email} />
        ) : (
          noCase(state)
        )}
      </div>
    </div>
  )
})

const _LoginInput = React.memo(
  ({
    className = '',
    onEmailSent,
    email,
    setEmail,
  }: {
    className?: string
    email: string
    setEmail: Dispatch<string>
    onEmailSent: () => void
  }) => {
    const { defaultErrorHandlerForNetworkCall } =
      useDefaultErrorHandlerForNetworkCall()
    const [commitMutation, isRunning] =
      useMutation<RequireUserSendEmailMutation>(graphql`
        mutation RequireUserSendEmailMutation($input: SendSignInEmailInput!) {
          sendSignInEmail(input: $input) {
            __typename
            _
          }
        }
      `)

    const [validateEmail, setValidateEmail] = useState(false)
    const emailValidation = useMemo(
      () => API.SendSignInEmail.guards.email(email.trim()),
      [email],
    )

    const inputRef = useRef<HTMLInputElement>(null)

    const handleContinueWithEmail = () => {
      if (emailValidation.error) {
        setValidateEmail(true)
      } else {
        const url = new URL(window.location.href)
        const dest = `${url.pathname}${url.search}`
        commitMutation({
          variables: { input: { email: emailValidation.value, dest } },
          onCompleted: onEmailSent,
          onError: (e) => {
            defaultErrorHandlerForNetworkCall({
              e,
              toast: 'Something went wrong.',
            })
          },
        })
      }
    }

    return (
      <div className={`${className}`}>
        <h2 className="text-4xl font-bold">Login / Sign Up</h2>
        <div className="flex gap-x-2 items-stretch mt-20">
          <input
            ref={inputRef}
            type="email"
            value={email}
            className={`border w-full
            ${
              validateEmail && emailValidation.error
                ? 'border-red-400 text-red-700'
                : 'border-gray-400'
            } rounded-lg px-4 py-3 w-full`}
            autoComplete="email"
            placeholder="Email"
            onChange={(e) => {
              setEmail(e.target.value)
              setValidateEmail(false)
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                fGet(inputRef.current).blur()
                handleContinueWithEmail()
              }
            }}
          />
        </div>
        <div className="inline-flex flex-col justify-items-stretch gap-y-4 mt-6">
          <button
            className="relative btn-lg btn-dark"
            disabled={isRunning}
            onClick={handleContinueWithEmail}
          >
            <div
              className={` flex gap-x-2 items-center 
              ${isRunning ? 'invisible' : ''}`}
            >
              <FontAwesomeIcon className="text-xl" icon={faEnvelope} /> Continue
              with Email
            </div>
            {isRunning && <Spinner />}
          </button>
          {/* <h2 className="ml-4">Or</h2> */}

          <div
            className=" my-3 grid items-center "
            style={{ grid: 'auto/1fr auto 1fr' }}
          >
            <div className="border-t border-gray-500"></div>
            <h2 className="mx-4">or</h2>
            <div className="border-t border-gray-500"></div>
          </div>
          <button
            className="btn-lg btn-dark flex gap-x-2 items-center whitespace-nowrap"
            // eslint-disable-next-line @typescript-eslint/no-misused-promises
            onClick={async () => {
              const provider = new GoogleAuthProvider()
              await signInWithRedirect(getAuth(), provider)
            }}
          >
            <FontAwesomeIcon className="text-xl" icon={faGoogle} /> Continue
            with Google
          </button>
        </div>
        <div className="mt-10 flex justify-end">
          <h2 className="">
            Read our{' '}
            <Link className=" border-b border-gray-500" href="/privacy">
              privacy policy
            </Link>
          </h2>
        </div>
      </div>
    )
  },
)

const _LoginEmailSent = React.memo(
  ({ className = '', email }: { className?: string; email: string }) => {
    return (
      <div className={`${className} max-w-[500px]`}>
        <h2 className="text-4xl font-bold">Email Sent</h2>
        <p className="p-base mt-6">
          We have sent you an email at{' '}
          <span className="font-bold">{email}</span>. Please follow the link in
          the email to continue.
        </p>
      </div>
    )
  },
)
