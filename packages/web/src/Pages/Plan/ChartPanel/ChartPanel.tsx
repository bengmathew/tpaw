import {faArrowsV, faLongArrowAltRight} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import Link from 'next/link'
import React, {useEffect, useState} from 'react'
import Measure, {BoundingRect} from 'react-measure'
import {fGet} from '../../../Utils/Utils'
import {useSimulation} from '../../App/WithSimulation'
import {DistributionCanvasReact} from './Chart/DistributionCanvasReact'
import {LegacyDisplay} from './LegacyDisplay'

const maxYScaleWithCeiling = 1.25
const maxYScaleWithoutCeiling = 1

export const ChartPanel = React.memo(
  ({className = '', isPortrait}: {className?: string; isPortrait: boolean}) => {
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
      <div
        className={`${className} relative pb-2 grid`}
        style={{
          grid: isPortrait
            ? '"label" auto "plot" 1fr "buttons" auto/1fr'
            : '"label" auto "buttons" auto "plot" 1fr/1fr',
          gridArea: 'chart',
        }}
      >
        <div
          className={`${
            (bounds?.width ?? 0) > 700 ? 'mt-4' : 'pt-header mt-2 sm:mt-0'
          }`}
        >
          <h2 className="text-xl sm:text-2xl -mt-1 font-bold">
            Spending During Retirement
          </h2>
          <h2 className="text-[13px] lighten -mt-1">
            Results from simulating your retirement {tpawResult.args.numRuns}{' '}
            times.
          </h2>
        </div>
        <div
          className={`grid  text-lg 
          ${isPortrait ? ' items-end ' : 'mt-3 '}`}
          style={{
            gridArea: 'buttons',
            grid: isPortrait
              ? '"rescale tasks" auto/1fr auto'
              : '"tasks" auto "rescale" auto/auto',
          }}
        >
          <Link href="/tasks-for-this-year">
            <a
              className="flex items-center gap-x-2"
              style={{gridArea: 'tasks'}}
            >
              <h2 className="font-medium">Tasks for this year</h2>
              <FontAwesomeIcon icon={faLongArrowAltRight} />
            </a>
          </Link>

          <button
            className="flex items-center py-2  -mb-2 font-medium "
            onClick={handleRescale}
            style={{gridArea: 'rescale'}}
          >
            <FontAwesomeIcon className="mr-1 text-[13px]" icon={faArrowsV} />
            <span className="">Rescale</span>
          </button>
        </div>
        <Measure bounds onResize={({bounds}) => setBounds(fGet(bounds))}>
          {({measureRef}) => (
            <div
              className=" -mx-[8px] relative z-0"
              // className="h-[30vh] -mx-[8px] relative z-0"
              ref={measureRef}
              style={{gridArea: 'plot'}}
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

        <LegacyDisplay className={`absolute left-3 ${isPortrait?'top-[60px]':'top-[145px]'}`} />
      </div>
    )
  }
)
