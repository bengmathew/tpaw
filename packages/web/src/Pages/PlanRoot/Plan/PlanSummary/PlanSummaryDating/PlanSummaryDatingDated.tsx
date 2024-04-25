import clsx from 'clsx'
import React, { useMemo } from 'react'
import { NormalizedDatingInfo } from '../../../../../UseSimulator/NormalizePlanParams/NormalizePlanParams'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'
import { useIANATimezoneName } from '../../../PlanRootHelpers/WithNonPlanParams'
import { DateTime } from 'luxon'
import { formatDistance } from 'date-fns'

export const PlanSummaryDatingDated = React.memo(
  ({
    className,
    datingInfo,
  }: {
    className?: string
    datingInfo: Extract<NormalizedDatingInfo, { isDated: true }>
  }) => {
    const { getZonedTime } = useIANATimezoneName()
    const { planParamsNorm } = useSimulation()

    const dateStr = useMemo(() => {
      const currentDateTime = getZonedTime(datingInfo.nowAsTimestamp)
      const lastUpdatedDateTime = getZonedTime(planParamsNorm.timestamp)
      return {
        currentTime: currentDateTime.toLocaleString(DateTime.DATETIME_MED),
        lastUpdated: lastUpdatedDateTime.toLocaleString(DateTime.DATETIME_MED),
        timeSinceLastUpdate: formatDistance(
          planParamsNorm.timestamp,
          datingInfo.nowAsTimestamp,
          { addSuffix: true },
        ),
      }
    }, [datingInfo, getZonedTime, planParamsNorm.timestamp])

    return (
      <div className={clsx(className)}>
        <div className="inline-grid gap-x-2" style={{ grid: 'auto/auto auto' }}>
          <h2 className="">Current Time:</h2>
          <h2 className="">{dateStr.currentTime}</h2>
          <h2 className="">Last Updated:</h2>
          <h2 className="">
            {dateStr.lastUpdated}{' '}
            <span className="">{dateStr.timeSinceLastUpdate}</span>
          </h2>
        </div>
      </div>
    )
  },
)
