import { fGet } from '@tpaw/common'
import { useFragment } from 'react-relay'
import { graphql } from 'relay-runtime'
import { createContext } from '../../Utils/CreateContext'
import { FirebaseUser } from '../App/WithFirebaseUser'
import { UserFragment_query$key } from './__generated__/UserFragment_query.graphql'
import {
  UserFragment_user$data,
  UserFragment_user$key,
} from './__generated__/UserFragment_user.graphql'

const userFragmentOnQueryGQL = graphql`
  fragment UserFragment_query on Query {
    user(userId: $userId) @include(if: $includeUser) {
      ...UserFragment_user
    }
  }
`

const userFragmentOnUserGQL = graphql`
  fragment UserFragment_user on User {
    id
    plan {
      id
      createdAt
      modifiedAt
      params
    }
  }
`

type _Value = UserFragment_query$key | null
const [UserFragmentContext, useUserFragment] =
  createContext<_Value>('FirebaseAuth')

export function useUser(firebaseUser: null): null
export function useUser(firebaseUser: FirebaseUser): UserFragment_user$data
export function useUser(
  firebaseUser: FirebaseUser | null,
): UserFragment_user$data | null {
  const userOnQuery = useFragment(userFragmentOnQueryGQL, useUserFragment())
  const user = useFragment<UserFragment_user$key>(
    userFragmentOnUserGQL,
    userOnQuery?.user ?? null,
  )
  return firebaseUser ? fGet(user) : null
}
export const WithUser = UserFragmentContext.Provider
