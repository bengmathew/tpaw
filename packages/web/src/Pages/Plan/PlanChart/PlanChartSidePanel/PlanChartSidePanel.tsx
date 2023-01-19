import React from 'react'
import {
  insetCSSStyle,
  InsetExt,
  newPadding,
  paddingCSSStyle,
  rectExt,
} from '../../../../Utils/Geometry'
import { PlanSectionName } from '../../PlanInput/Helpers/PlanSectionName'
import { PlanChartSizing } from '../PlanChart'
import { PlanChartSidePanelLegacyCard } from './PlanChartSidePanelLegacyCard'
import { PlanChartSidePanelTasksCard } from './PlanChartSidePanelTasksCard'

type _DynamicSizing = { inset: InsetExt }

export const PlanChartSidePanel = React.memo(
  ({
    targetDynamicSizing,
    fixedSizing: { cardPadding, intraGap },
    duration,
    layout,
    section,
  }: {
    targetDynamicSizing: _DynamicSizing
    fixedSizing: PlanChartSizing['fixed']
    duration: number
    layout: 'laptop' | 'desktop' | 'mobile'
    section: PlanSectionName
  }) => {
    return (
      <div
        className="absolute overflow-hidden"
        style={{
          transitionProperty: 'right, bottom, top, left',
          transitionDuration: `${duration}ms`,
          // Don't use transform: translate(x, y) here because right edge
          // movement is not linear.
          ...insetCSSStyle(targetDynamicSizing.inset),
        }}
      >
        <PlanChartSidePanelLegacyCard
          className=" bg-cardBG rounded-2xl "
          style={paddingCSSStyle(cardPadding)}
          layout={layout}
        />
        <PlanChartSidePanelTasksCard
          layout={layout}
          className="rounded-2xl "
          style={{
            ...paddingCSSStyle(cardPadding),
            marginTop: `${intraGap}px`,
          }}
          cardPadding={cardPadding}
          section={section}
        />
      </div>
    )
  },
)

// const _transformSizing = ({
//   sidePanel,
// }: PlanChartSidePanelSizing['dynamic']['dialogInput']) => {
//   const { inset } = sidePanel
//   const chart = {
//     position: rectExt({
//       x: 0,
//       y: 0,
//       height: inset.height,
//       width: inset.width,
//     }),
//     padding: newPadding({ horz: 0, top: 60, bottom: 10 }),
//   }
//   return { ...sidePanel, chart }
// }
