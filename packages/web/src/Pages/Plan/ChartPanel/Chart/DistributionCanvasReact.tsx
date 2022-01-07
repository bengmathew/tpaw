import React, {useEffect, useRef} from 'react'
import { UseTPAWWorkerResult } from '../../../../TPAWSimulator/Worker/UseTPAWWorker'
import { fGet } from '../../../../Utils/Utils'
import {DistributionCanvas} from './DistributionCanvas'

export const DistributionCanvasReact = React.memo(
  ({
    size: {width, height},
    tpawResult: tpawResult,
    highlightPercentiles,
    maxY,
  }: {
    size: {width: number; height: number}
    tpawResult: UseTPAWWorkerResult
    highlightPercentiles: number[]
    maxY: number
  }) => {
    const chartRef = useRef<DistributionCanvas | null>(null)

    useEffect(() => {
      fGet(chartRef.current).setSize({width, height})
    }, [width, height])

    useEffect(() => {
      fGet(chartRef.current).setData(tpawResult)
    }, [tpawResult])

    useEffect(() => {
      fGet(chartRef.current).setMaxY(maxY)
    }, [maxY])

    return (
      // select-none because otherwise moving the chart pointer was selecting text
      // on mobile. Seems to work only on Android though.
      <div className=" select-none ">
        <canvas
          style={{touchAction: 'none'}}
          ref={canvas => {
            if (canvas && !chartRef.current) {
              chartRef.current = new DistributionCanvas(
                canvas,
                {width, height},
                tpawResult,
                highlightPercentiles,
                maxY
              )
            }
          }}
        />
      </div>
    )
  }
)
