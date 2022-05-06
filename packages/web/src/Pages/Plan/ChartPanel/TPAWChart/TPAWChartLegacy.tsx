import React, {useMemo} from 'react'
import {
  ChartReact,
  ChartReactSizing,
  ChartReactState,
  ChartReactStatefull,
} from '../../../Common/Chart/ChartReact'
import {chartDrawLegacy} from '../LegacyComponents/ChartDrawLegacy'
import {TPAWChartDataLegacy} from './TPAWChartDataLegacy'

export const TPAWChartLegacy = React.memo(
  React.forwardRef(
    (
      {
        state,
        starting,
      }: {
        state: ChartReactState<TPAWChartDataLegacy>
        starting: {sizing: ChartReactSizing}
      },
      ref: React.ForwardedRef<ChartReactStatefull>
    ) => {
      const components = useMemo(() => {
        const legacy = chartDrawLegacy()
        return {
          arr: [legacy],
          byName: {legacy},
        }
      }, [])

      return (
        <ChartReact<TPAWChartDataLegacy>
          ref={ref}
          state={state}
          starting={starting}
          components={components.arr}
        />
      )
    }
  )
)

