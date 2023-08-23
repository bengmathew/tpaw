import { fGet } from '@tpaw/common'
import Link from 'next/link'
import React from 'react'
import { graphql, useLazyLoadQuery } from 'react-relay'
import { appPaths } from '../../AppPaths'
import { AppPage } from '../App/AppPage'
import { useUserGQLArgs } from '../App/WithFirebaseUser'
import { WithUser, useUser } from '../App/WithUser'
import { AccountQuery } from './__generated__/AccountQuery.graphql'

const query = graphql`
  query AccountQuery($userId: ID!, $includeUser: Boolean!) {
    ...WithUser_query
  }
`

export const Account = React.memo(() => {
  const userGQLArgs = useUserGQLArgs()
  const data = useLazyLoadQuery<AccountQuery>(query, { ...userGQLArgs })

  return (
    <WithUser userFragmentOnQueryKey={data}>
      <_Account />
    </WithUser>
  )
})

const _Account = React.memo(() => {
  const user = fGet(useUser())
  return (
    <AppPage
      className="pt-header"
      title={'Account - TPAW Planner'}
    >
      <div className="flex flex-col items-center mb-20 mt-6">
        <div className="w-full max-w-[650px] px-4 z-0">
          <div className=" ">
            <h1 className="font-bold text-4xl">Account</h1>
            <div className="mt-4">
              <h2 className="">
                You are logged in as {fGet(user.firebaseUser.email)}
              </h2>
              <div className="mt-4 flex flex-col gap-y-2 items-start">
                <Link
                  className="block text-xl underline py-1"
                  href={appPaths.logout()}
                >
                  Logout
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </AppPage>
  )
})
