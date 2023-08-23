import { useFragment } from 'react-relay'
import { graphql } from 'relay-runtime'
import { PlanWithoutParamsFragment$key } from './__generated__/PlanWithoutParamsFragment.graphql'

export const usePlanWithoutParamsFragment = (key: PlanWithoutParamsFragment$key) =>
  useFragment(
    graphql`
      fragment PlanWithoutParamsFragment on PlanWithHistory {
        id
        isMain
        label
        slug
        addedToServerAt
        sortTime
        lastSyncAt
      }
    `,
    key,
  )
