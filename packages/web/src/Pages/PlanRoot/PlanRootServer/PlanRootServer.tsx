import { assert } from '@tpaw/common'
import React, { useEffect, useLayoutEffect } from 'react'
import { graphql, useLazyLoadQuery } from 'react-relay'
import { appPaths } from '../../../AppPaths'
import { useUserGQLArgs } from '../../App/WithFirebaseUser'
import { WithUser } from '../../App/WithUser'
import { TARGET_UNDO_DEPTH } from '../PlanRootHelpers/UseWorkingPlan'
import { PlanServer } from '../PlanServerImpl/PlanServerImpl'
import { PlanRootServerQuery } from './__generated__/PlanRootServerQuery.graphql'

export const PlanRootServer = React.memo(
  ({
    src,
  }: {
    src: { type: 'serverMain' } | { type: 'serverAlt'; slug: string }
  }) => {
    const userGQLArgs = useUserGQLArgs()

    const data = useLazyLoadQuery<PlanRootServerQuery>(
      graphql`
        query PlanRootServerQuery(
          $userId: ID!
          $includeUser: Boolean!
          $targetCount: Int!
          $slug: String
        ) {
          ...WithUser_query

          user(userId: $userId) @include(if: $includeUser) {
            plan(slug: $slug) {
              id
              isMain
              label
              slug
              addedToServerAt
              sortTime
              lastSyncAt
              reverseHeadIndex
              planParamsPostBase(targetCount: $targetCount) {
                id
                params
                change
              }
            }
          }
        }
      `,
      {
        ...userGQLArgs,
        targetCount: TARGET_UNDO_DEPTH + 1,
        slug: src.type === 'serverAlt' ? src.slug : null,
      },
      // Don't use the cache because we don't update it when syncing, so it will
      // be stale. Note this will result in a double network call in dev
      // due to recreating the component for strict mode.
      { fetchPolicy: 'network-only' },
    )
    assert(data.user)

      
    return (
      <WithUser userFragmentOnQueryKey={data}>
        <PlanServer
          plan={data.user.plan}
          planPaths={
            src.type === 'serverMain' ? appPaths.plan : appPaths['alt-plan']
          }
        />
      </WithUser>
    )
  },
)
