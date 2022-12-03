import React, { useMemo } from 'react'
import {
  insetCSSStyle,
  InsetExt,
  newPadding,
  paddingCSSStyle,
  rectExt,
} from '../../../../Utils/Geometry'
import { PlanChartSizing, PlanChartTransitionState } from '../PlanChart'
import { PlanChartSidePanelLegacyCard } from './PlanChartSidePanelLegacyCard'
import { PlanChartSidePanelTasksCard } from './PlanChartSidePanelTasksCard'

type PlanChartSidePanelSizing = {
  dynamic: Record<
    PlanChartTransitionState,
    {
      sidePanel: {
        inset: InsetExt
      }
    }
  >
  fixed: PlanChartSizing['fixed']
}
export const PlanChartSidePanel = React.memo(
  ({
    sizing,
    transition,
    layout,
  }: {
    sizing: PlanChartSidePanelSizing
    transition: { target: PlanChartTransitionState; duration: number }
    layout: 'laptop' | 'desktop' | 'mobile'
  }) => {
    const targetSizing = useMemo(
      () => _transformSizing(sizing.dynamic[transition.target]),
      [sizing.dynamic, transition.target],
    )

    const { cardPadding, intraGap } = sizing.fixed

    return (
      <div
        className="absolute overflow-hidden"
        style={{
          transitionProperty: 'right, bottom, top, left',
          transitionDuration: `${transition.duration}ms`,
          // Don't use transform: translate(x, y) here because right edge
          // movement is not linear.
          ...insetCSSStyle(targetSizing.inset),
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
        />
      </div>
    )
  },
)

const _transformSizing = ({
  sidePanel,
}: PlanChartSidePanelSizing['dynamic']['hidden']) => {
  const { inset } = sidePanel
  const chart = {
    position: rectExt({
      x: 0,
      y: 0,
      height: inset.height,
      width: inset.width,
    }),
    padding: newPadding({ horz: 0, top: 60, bottom: 10 }),
  }
  return { ...sidePanel, chart }
}
