import { assert } from '@tpaw/common'
import React, { useState } from 'react'
import { graphql, useLazyLoadQuery } from 'react-relay'
import { AppError } from '../../App/AppError'
import { useUserGQLArgs } from '../../App/WithFirebaseUser'
import { TARGET_UNDO_DEPTH } from '../PlanRootHelpers/UseWorkingPlan'
import { SimulationParams } from '../PlanRootHelpers/WithSimulation'
import { PlanRootServerQuery } from './__generated__/PlanRootServerQuery.graphql'
import { WithUser } from '../../App/WithUser'
import { PlanServerImpl } from '../PlanServerImpl/PlanServerImpl'
import { appPaths } from '../../../AppPaths'
import * as uuid from 'uuid'

export const PlanRootServer = React.memo(
  ({
    src,
    pdfReportInfo,
  }: {
    src: { type: 'serverMain' } | { type: 'serverAlt'; slug: string }
    pdfReportInfo: SimulationParams['pdfReportInfo']
  }) => {
    const [key, setKey] = React.useState(0)
    return (
      <_Body
        key={key}
        src={src}
        pdfReportInfo={pdfReportInfo}
        reload={() => setKey(key + 1)}
      />
    )
  },
)
const _Body = React.memo(
  ({
    src,
    pdfReportInfo,
    reload,
  }: {
    src: { type: 'serverMain' } | { type: 'serverAlt'; slug: string }
    pdfReportInfo: SimulationParams['pdfReportInfo']
    reload: () => void
  }) => {
    const userGQLArgs = useUserGQLArgs()

    // Random fetchKey will enforce a refetch on component reload. This is
    // necessary because network-only fetchPolicy does not seem to work
    // https://github.com/facebook/relay/issues/3502.
    const [fetchKey] = useState(() => uuid.v4())

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
              isDated
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
      // Note: network-only might not be working as expected: https://github.com/facebook/relay/issues/3502.
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
          reload={reload}
        />
      </WithUser>
    )
  },
)
