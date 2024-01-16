import { FirebaseError } from 'firebase/app'
import {
  getAuth,
  isSignInWithEmailLink,
  signInWithEmailLink,
} from 'firebase/auth'
import React, { useEffect, useState } from 'react'
import { appPaths } from '../../AppPaths'
import { asyncEffect } from '../../Utils/AsyncEffect'
import { useURLParam } from '../../Utils/UseURLParam'
import { useURLUpdater } from '../../Utils/UseURLUpdater'
import { useFirebaseUser } from '../App/WithFirebaseUser'
import { assert } from '../../Utils/Utils'

type _State =
  | { type: 'error'; message: string }
  | { type: 'doSignIn'; email: string }
  | { type: 'loggedIn' }

export const Email = React.memo(() => {
  const auth = getAuth()
  const isLink = isSignInWithEmailLink(auth, window.location.href)
  const email = window.localStorage.getItem('emailForSignIn')
  const isLoggedIn = useFirebaseUser() !== null

  const [state, setState] = useState<_State>(
    isLoggedIn
      ? { type: 'loggedIn' }
      : !isLink
        ? { type: 'error', message: 'This link is no longer valid.' }
        : !email
          ? {
              type: 'error',
              message:
                'Device changed. Please open the link on the same device.',
            }
          : { type: 'doSignIn', email },
  )

  const urlUpdater = useURLUpdater()

  const dest = useURLParam('dest') ?? appPaths.plan()
  useEffect(() => {
    if (isLoggedIn) urlUpdater.replace(dest)
  }, [dest, isLoggedIn, urlUpdater])

  useEffect(() => {
    return asyncEffect(async () => {
      if (state.type !== 'doSignIn') return
      try {
        await signInWithEmailLink(getAuth(), state.email, window.location.href)
      } catch (e) {
        assert(e !== null)
        if (e instanceof FirebaseError) {
          if (e.code === 'auth/email-already-in-use') {
            // This is not an error but sometimes happens when we manually
            // delete a user in firebase and then is recreated.
          } else if (e.code === 'auth/id-token-expired') {
            setState({
              type: 'error',
              message: 'This link is no longer valid.',
            })
          } else {
            throw e
          }
        } else {
          throw e
        }
      }
      window.localStorage.removeItem('emailForSignIn')
    })
  }, [state])

  switch (state.type) {
    case 'loggedIn':
    case 'doSignIn':
      return <></>
    case 'error':
      return (
        <div className="page flex flex-col justify-center items-center h-screen">
          <div className="">
            <h2 className="font-bold text-3xl">Error</h2>
            <h2 className="mt-6">{state.message}</h2>
          </div>
        </div>
      )
  }
})
