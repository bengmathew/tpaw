import React from 'react'
import { PlanRoot } from '../../src/Pages/PlanRoot/PlanRoot'
import { planRootGetStaticPaths } from '../../src/Pages/PlanRoot/PlanRootHelpers/PlanRootGetStaticPaths'
import {
  planRootGetStaticProps,
  PlanStaticProps,
} from '../../src/Pages/PlanRoot/PlanRootHelpers/PlanRootGetStaticProps'

export default React.memo(({ planContent }: PlanStaticProps) => {
  return <PlanRoot planContent={planContent} src={{ type: 'file' }} />
})

export const getStaticProps = planRootGetStaticProps
export const getStaticPaths = planRootGetStaticPaths
