import { assert, fGet } from '@tpaw/common'
import React, { ReactNode, useMemo } from 'react'
import { useFragment } from 'react-relay'
import { graphql } from 'relay-runtime'
import { createContext } from '../../Utils/CreateContext'
import { FirebaseUser, useFirebaseUser } from './WithFirebaseUser'
import { WithNonPlanParams } from '../PlanRoot/PlanRootHelpers/WithNonPlanParams'
import { WithUser_query$key } from './__generated__/WithUser_query.graphql'
import {
  WithUser_user$data,
  WithUser_user$key,
} from './__generated__/WithUser_user.graphql'

export type User = WithUser_user$data & {
  firebaseUser: FirebaseUser
}
const [Context, useUser] = createContext<User | null>('User')
export { useUser }
export const WithUser = ({
  userFragmentOnQueryKey,
  children,
}: {
  userFragmentOnQueryKey: WithUser_query$key | null
  children: ReactNode
}) => {
  return userFragmentOnQueryKey ? (
    <_LoggedIn userFragmentOnQueryKey={userFragmentOnQueryKey}>
      <WithNonPlanParams>{children}</WithNonPlanParams>
    </_LoggedIn>
  ) : (
    <_NotLoggedIn>
      <WithNonPlanParams>{children}</WithNonPlanParams>
    </_NotLoggedIn>
  )
}

const _NotLoggedIn = React.memo(({ children }: { children: ReactNode }) => {
  return <Context.Provider value={null}>{children}</Context.Provider>
})

const _LoggedIn = React.memo(
  ({
    children,
    userFragmentOnQueryKey,
  }: {
    children: ReactNode
    userFragmentOnQueryKey: WithUser_query$key
  }) => {
    const firebaseUser = fGet(useFirebaseUser())
    const userOnQuery = useFragment(
      graphql`
        fragment WithUser_query on Query {
          user(userId: $userId) @include(if: $includeUser) {
            ...WithUser_user
          }
        }
      `,
      userFragmentOnQueryKey,
    )

    assert(userOnQuery.user)

    const userOnUser = useFragment<WithUser_user$key>(
      graphql`
        fragment WithUser_user on User {
          id
          plans {
            id
            label
            slug
            addedToServerAt
            sortTime
            lastSyncAt
            isMain
          }
          nonPlanParamsLastUpdatedAt
          nonPlanParams
        }
      `,
      userOnQuery.user,
    )

    const user = useMemo(() => {
      return { ...userOnUser, firebaseUser }
    }, [firebaseUser, userOnUser])

    return <Context.Provider value={user}>{children}</Context.Provider>
  },
)
