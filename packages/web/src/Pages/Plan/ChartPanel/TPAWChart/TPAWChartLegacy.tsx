import React, { useCallback } from 'react'
import {ChartXYRange} from '../../../Common/Chart/Chart'
import {
  ChartReact,
  ChartReactSizing,
  ChartReactStatefull,
} from '../../../Common/Chart/ChartReact'
import {chartDrawLegacy} from '../LegacyComponents/ChartDrawLegacy'
import {TPAWChartDataLegacy} from './TPAWChartDataLegacy'

export const TPAWChartLegacy = React.memo(
  React.forwardRef(
    (
      {
        starting,
      }: {
        starting: {
          data: TPAWChartDataLegacy
          xyRange: ChartXYRange
          sizing: ChartReactSizing
        }
      },
      ref: React.ForwardedRef<ChartReactStatefull<TPAWChartDataLegacy>>
    ) => {
      const components = useCallback(()=>[chartDrawLegacy()], [])
      return (
        <ChartReact<TPAWChartDataLegacy>
          ref={ref}
          starting={starting}
          components={components}
        />
      )
    }
  )
)

