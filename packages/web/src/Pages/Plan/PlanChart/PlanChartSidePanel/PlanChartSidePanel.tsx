import React from 'react'
import { InsetExt, paddingCSSStyle } from '../../../../Utils/Geometry'
import { PlanSectionName } from '../../PlanInput/Helpers/PlanSectionName'
import { PlanChartSizing } from '../PlanChart'
import { PlanChartSidePanelLegacyCard } from './PlanChartSidePanelLegacyCard'
import { PlanChartSidePanelMenu } from './PlanChartSidePanelMenu'
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
          transitionProperty: 'right, top, width, height',
          transitionDuration: `${duration}ms`,
          // Don't use transform: translate(x, y) here because right edge
          // movement is not linear.
          right: `${targetDynamicSizing.inset.right}px`,
          top: `${targetDynamicSizing.inset.top}px`,
          // Use width and height and not left and bottom to
          // keep size constant through transition.
          width: `${targetDynamicSizing.inset.width}px`,
          height: `${targetDynamicSizing.inset.height}px`,
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
        <PlanChartSidePanelMenu
          className="bg-cardBG rounded-2xl"
          style={{
            ...paddingCSSStyle(cardPadding),
            marginTop: `${intraGap}px`,
          }}
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
