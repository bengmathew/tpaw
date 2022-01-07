import _ from 'lodash'
import React, { useMemo } from 'react'
import { formatCurrency } from '../../../Utils/FormatCurrency'
import { useSimulation } from '../../App/WithSimulation'

export const LegacyDisplay = React.memo(
  ({className = ''}: {className?: string}) => {
    const {params, tpawResult} = useSimulation()
    const legacyByPercentile = useMemo(
      () =>
        _.sortBy(
          tpawResult.legacyByPercentile.filter(x =>
            [5, 50, 95].includes(x.percentile)
          ),
          x => -x.percentile
        ),
      [tpawResult.legacyByPercentile]
    )

    if (params.legacy.total === 0 && params.spendingCeiling === null)
      return <></>

    return (
      <div
        className={`${className}  bg-gray-100  border border-gray-400 rounded-xl  flex  `}
      >
        <div className=" rounded-lg px-2 py-1">
          <div className="font-semibold sm:font-medium text-sm sm:text-base w-full flex items-center justify-between gap-x-2">
            Legacy
          </div>

          <div className="">
            <div
              className={`grid text-[11px] sm:text-[12px] gap-x-4 lighten `}
              style={{grid: 'auto / auto 1fr'}}
            >
              {legacyByPercentile.map(({legacy, percentile}) => (
                <React.Fragment key={percentile}>
                  <h2 className="text-left">{`${percentile}th`}</h2>
                  <h2 className="text-right">{`${formatCurrency(legacy)}`}</h2>
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>
      </div>
    )
  }
)
