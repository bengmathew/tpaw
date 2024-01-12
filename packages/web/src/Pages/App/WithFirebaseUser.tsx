import * as Sentry from '@sentry/nextjs'
import { fGet } from '@tpaw/common'
import { initializeApp } from 'firebase/app'
import { getAuth, onAuthStateChanged, User } from 'firebase/auth'
import React, { ReactNode, useEffect, useState } from 'react'
import { createContext } from '../../Utils/CreateContext'
import { Config } from '../Config'

export type FirebaseUser = User
initializeApp({
  apiKey: Config.client.google.firebase.apiKey,
  authDomain: Config.client.google.firebase.authDomain,
  projectId: Config.client.google.cloud.projectId,
  storageBucket: Config.client.google.firebase.storageBucket,
  messagingSenderId: Config.client.google.firebase.messagingSenderId,
  appId: Config.client.google.firebase.appId,
})

const [Context, useFirebaseUser] = createContext<FirebaseUser | null>(
  'FirebaseAuth',
)
export { useFirebaseUser }

export const useUserGQLArgs = () => {
  const firebaseUser = useFirebaseUser()
  return firebaseUser
    ? { userId: firebaseUser.uid, includeUser: true }
    : { userId: '', includeUser: false }
}

type _State =
  | { initialized: false }
  | { initialized: true; user: FirebaseUser | null }

export const WithFirebaseUser = React.memo(
  ({ children }: { children: ReactNode }) => {
    const [state, setState] = useState<_State>({ initialized: false })

    useEffect(
      () =>
        onAuthStateChanged(getAuth(), (user) => {
          Sentry.setUser(
            user ? { id: user.uid, email: user.email ?? undefined } : null,
          )
          if (Config.client.google.analytics.tagId) {
            fGet((window as any).gtag)(
              'config',
              Config.client.google.analytics.tagId,
              { user_id: user?.uid ?? null },
            )
          }
          setState({ initialized: true, user })
        }),
      [],
    )

    if (!state.initialized) return <></>
    return (
      <Context.Provider key={state.user?.uid ?? 'none'} value={state.user}>
        {children}
      </Context.Provider>
    )
  },
)
