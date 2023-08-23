import { MarketData, noCase } from '@tpaw/common'
import React from 'react'
import { PlanContent } from './PlanRootHelpers/PlanRootGetStaticProps'
import { WithMarketData } from './PlanRootHelpers/WithMarketData'
import { WithPlanContent } from './PlanRootHelpers/WithPlanContent'
import { WithWASM } from './PlanRootHelpers/WithWASM'
import { PlanRootLink } from './PlanRootLink/PlanRootLink'
import { PlanRootLocalMain } from './PlanRootLocalMain/PlanRootLocalMain'
import { PlanRootServer } from './PlanRootServer/PlanRootServer'

export const PlanRoot = React.memo(
  ({
    planContent,
    marketData,
    src,
  }: {
    planContent: PlanContent
    marketData: MarketData.Data
    src:
      | { type: 'serverMain' }
      | { type: 'localMain' }
      | { type: 'serverAlt'; slug: string }
      | { type: 'link' }
  }) => {
    return (
      <WithWASM>
        <WithPlanContent planContent={planContent}>
          <WithMarketData marketData={marketData}>
            {src.type === 'serverMain' ? (
              <PlanRootServer key="main" src={src} />
            ) : src.type === 'serverAlt' ? (
              <PlanRootServer key={src.slug} src={src} />
            ) : src.type === 'localMain' ? (
              <PlanRootLocalMain />
            ) : src.type === 'link' ? (
              <PlanRootLink />
            ) : (
              noCase(src)
            )}
          </WithMarketData>
        </WithPlanContent>
      </WithWASM>
    )
  },
)
