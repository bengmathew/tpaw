import _ from 'lodash'
import React, {useMemo} from 'react'
import {TPAWParams} from '../../TPAWSimulator/TPAWParams'
import {UseTPAWWorkerResult} from '../../TPAWSimulator/Worker/UseTPAWWorker'
import {formatCurrency} from '../../Utils/FormatCurrency'
import {StateObj} from '../../Utils/UseStateObj'

export const LegacyDisplay = React.memo(
  ({
    className = '',
    tpawResult,
    params: paramsObj,
  }: {
    className?: string
    params: StateObj<TPAWParams>
    tpawResult: UseTPAWWorkerResult
  }) => {
    const {value: params} = paramsObj
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
        className={`${className} absolute bg-gray-100  border border-gray-400 rounded-xl  flex left-[15px] top-[30px]`}
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
