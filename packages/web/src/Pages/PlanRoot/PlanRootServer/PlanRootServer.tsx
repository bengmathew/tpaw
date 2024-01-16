import { assert } from '@tpaw/common'
import React from 'react'
import { graphql, useLazyLoadQuery } from 'react-relay'
import { appPaths } from '../../../AppPaths'
import { AppError } from '../../App/AppError'
import { useUserGQLArgs } from '../../App/WithFirebaseUser'
import { WithUser } from '../../App/WithUser'
import { TARGET_UNDO_DEPTH } from '../PlanRootHelpers/UseWorkingPlan'
import { SimulationParams } from '../PlanRootHelpers/WithSimulation'
import { PlanServerImpl } from '../PlanServerImpl/PlanServerImpl'
import { PlanRootServerQuery } from './__generated__/PlanRootServerQuery.graphql'

export const PlanRootServer = React.memo(
  ({
    src,
    pdfReportInfo,
  }: {
    src: { type: 'serverMain' } | { type: 'serverAlt'; slug: string }
    pdfReportInfo: SimulationParams['pdfReportInfo']
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
    if (!data.user.plan) {
      assert(src.type === 'serverAlt')
      throw new AppError('404')
    }

    return (
      <WithUser userFragmentOnQueryKey={data}>
        <PlanServerImpl
          plan={data.user.plan}
          planPaths={
            src.type === 'serverMain' ? appPaths.plan : appPaths['alt-plan']
          }
          pdfReportInfo={pdfReportInfo}
        />
      </WithUser>
    )
  },
)
