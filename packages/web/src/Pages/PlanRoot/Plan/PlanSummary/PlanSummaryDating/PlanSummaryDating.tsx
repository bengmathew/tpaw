import clsx from 'clsx'
import React from 'react'
import { useSimulationInfo } from '../../../PlanRootHelpers/WithSimulation'
import { usePlanColors } from '../../UsePlanColors'
import { PlanSummaryDatingDated } from './PlanSummaryDatingDated'
import { PlanSummaryDatingDateless } from './PlanSummaryDatingDateless'

export const PlanSummaryDating = React.memo(
  ({ className }: { className?: string }) => {
    const { planParamsNormInstant } = useSimulationInfo()

    if (planParamsNormInstant.datingInfo.isDated) return <></>

    return (
      <div
        className={clsx(
          className,
          'border-[1px border-gray-400  rounded-md p-2 mt-4 -mb-10 ',
        )}
      >
        {planParamsNormInstant.datingInfo.isDated ? (
          <PlanSummaryDatingDated
            className=""
            datingInfo={planParamsNormInstant.datingInfo}
          />
        ) : (
          <PlanSummaryDatingDateless
            className=""
            datingInfo={planParamsNormInstant.datingInfo}
          />
        )}
      </div>
    )
  },
)
