import React from 'react'
import { RequireUser } from '../../src/Pages/App/RequireUser'
import { PlanRoot } from '../../src/Pages/PlanRoot/PlanRoot'
import { planRootGetStaticPaths } from '../../src/Pages/PlanRoot/PlanRootHelpers/PlanRootGetStaticPaths'
import {
  PlanStaticProps,
  planRootGetStaticProps,
} from '../../src/Pages/PlanRoot/PlanRootHelpers/PlanRootGetStaticProps'
import { useURLParam } from '../../src/Utils/UseURLParam'
import { AppError } from '../../src/Pages/App/AppError'

// FEATURE: See if we can replace ref query param with /[plan] route. The
// reason we don't have it as a query params is because we can't generate
// getStaticPaths for the route.
export default React.memo(({ planContent, marketData }: PlanStaticProps) => {
  const slug = useURLParam('plan')
  if (!slug) throw new AppError('404')
  return (
    <RequireUser>
      <PlanRoot
        planContent={planContent}
        marketData={marketData}
        src={{ type: 'serverAlt', slug }}
      />
    </RequireUser>
  )
})

export const getStaticProps = planRootGetStaticProps
export const getStaticPaths = planRootGetStaticPaths
