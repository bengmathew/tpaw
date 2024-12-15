import clsx from 'clsx'
import React, { useMemo } from 'react'
import { NormalizedDatingInfo } from '../../../../../Simulator/NormalizePlanParams/NormalizePlanParams'
import { useSimulationInfo } from '../../../PlanRootHelpers/WithSimulation'
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
    const { planParamsNormInstant } = useSimulationInfo()

    const dateStr = useMemo(() => {
      const currentDateTime = getZonedTime(datingInfo.nowAsTimestamp)
      const lastUpdatedDateTime = getZonedTime(planParamsNormInstant.timestamp)
      return {
        currentTime: currentDateTime.toLocaleString(DateTime.DATETIME_MED),
        lastUpdated: lastUpdatedDateTime.toLocaleString(DateTime.DATETIME_MED),
        timeSinceLastUpdate: formatDistance(
          planParamsNormInstant.timestamp,
          datingInfo.nowAsTimestamp,
          { addSuffix: true },
        ),
      }
    }, [datingInfo, getZonedTime, planParamsNormInstant.timestamp])

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
