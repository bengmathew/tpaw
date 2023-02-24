import { fGet } from '@tpaw/common'
import React, { useLayoutEffect, useRef, useState } from 'react'
import { useChartData } from './App/WithChartData'
import { ChartReactStatefull } from './Common/Chart/ChartReact'
import { TPAWChartDataMain } from './Plan/PlanChart/TPAWChart/TPAWChartDataMain'

export const Print = React.memo(
  ({ className = '' }: { className?: string }) => {
    const [containerDiv, setContainerDiv] = useState<HTMLElement | null>(null)
    const chartRef = useRef<ChartReactStatefull<TPAWChartDataMain> | null>(null)

    const allChartData = useChartData()
    const chartMainData = fGet(
      allChartData.byYearsFromNowPercentiles.get('spending-total'),
    )

    const [width, setWidth] = useState(0)

    useLayoutEffect(() => {
      if (!containerDiv) return
      const rs = new ResizeObserver(() => {
        const width = containerDiv.getBoundingClientRect().width
        setWidth(width)
        // chartRef.current?.setSizing(
        //   {
        //     position: rectExt({ x: 0, y: 0, width, height: 500 }),
        //     padding: newPadding({ horz: 0, bottom: 30 }),
        //   },
        //   null,
        // )
      })
      rs.observe(containerDiv, { box: 'border-box' })
      return () => {
        rs.disconnect()
      }
    }, [containerDiv])

    return (
      <div
        className={`page ${className} mx-[2cm] my-[2cm]`}
        ref={setContainerDiv}
      >
        <h1 className="font-bold text-4xl"> TPAW</h1>
        <div className="relative" style={{ height: '500px' }}></div>
      </div>
    )
  },
)

// const _padding = () => ({
//   top: 20, // 20 is needed for the max y label.
//   bottom:
//     4 + // line at bottom of plotarea.
//     xAxisSizing.height +
//     (hasPartner
//       ? +3 * xAxisSizing.gap + xAxisSizing.height
//       : 2 * xAxisSizing.gap),
// })
