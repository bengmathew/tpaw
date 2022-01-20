import { faArrowsV, faLongArrowAltRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import Link from 'next/link'
import React, { useEffect, useState } from 'react'
import Measure, { BoundingRect } from 'react-measure'
import { fGet } from '../../../Utils/Utils'
import { useSimulation } from '../../App/WithSimulation'
import { DistributionCanvasReact } from './Chart/DistributionCanvasReact'
import { LegacyDisplay } from './LegacyDisplay'

const maxYScaleWithCeiling = 1.25
const maxYScaleWithoutCeiling = 1

export const ChartPanel = React.memo(
  ({className = ''}: {className?: string}) => {
    const {params, tpawResult, highlightPercentiles} = useSimulation()
    const maxYScale =
      params.spendingCeiling === null
        ? maxYScaleWithoutCeiling
        : maxYScaleWithCeiling

    const [bounds, setBounds] = useState<BoundingRect | null>(null)

    const [maxY, setMaxY] = useState(0)
    useEffect(() => {
      if (maxY === 0 && tpawResult)
        setMaxY(Math.max(1, tpawResult.maxWithdrawal * maxYScale))
    }, [maxY, maxYScale, tpawResult])
    const handleRescale = () =>
      setMaxY(Math.max(1, tpawResult.maxWithdrawal * maxYScale))

    return (
      <div className={`${className} relative pb-2 `}>
        <div className="pt-2">
          <h2 className="text-xl -mt-1 font-bold">
            Spending During Retirement
          </h2>
          <h2 className="text-[13px] lighten -mt-1">
            Results from simulating your retirement {tpawResult.args.numRuns}{' '}
            times.
          </h2>
        </div>
        <LegacyDisplay className="absolute top-[55px]" />
        <Measure bounds onResize={({bounds}) => setBounds(fGet(bounds))}>
          {({measureRef}) => (
            <div
              className="h-[200px] sm:h-[225px] md:h-[300px] lg:h-[375px] -mx-1 relative z-0"
              ref={measureRef}
            >
              <div className=" absolute">
                {bounds !== null && (
                  <DistributionCanvasReact
                    {...{tpawResult, maxY, highlightPercentiles}}
                    size={bounds}
                  />
                )}
              </div>
            </div>
          )}
        </Measure>
        <div className="flex justify-between items-end text-lg">
          <button
            className="flex items-center py-2  -mb-2 font-medium "
            onClick={handleRescale}
          >
            <FontAwesomeIcon className="mr-1 text-[13px]" icon={faArrowsV} />
            <span className="">Rescale</span>
          </button>
          <Link href="/tasks-for-this-year">
            <a className="flex items-center gap-x-2">
              <h2 className="font-medium">Tasks for this year</h2>
              <FontAwesomeIcon icon={faLongArrowAltRight} />
            </a>
          </Link>
        </div>
      </div>
    )
  }
)
