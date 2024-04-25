import clsx from 'clsx'
import React from 'react'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'
import { usePlanColors } from '../../UsePlanColors'
import { PlanSummaryDatingDated } from './PlanSummaryDatingDated'
import { PlanSummaryDatingDateless } from './PlanSummaryDatingDateless'

export const PlanSummaryDating = React.memo(
  ({ className }: { className?: string }) => {
    const { planParamsNorm } = useSimulation()

    if (planParamsNorm.datingInfo.isDated) return <></>

    return (
      <div
        className={clsx(
          className,
          'border-[1px border-gray-400  rounded-md p-2 mt-4 -mb-10 ',
        )}
      >
        {planParamsNorm.datingInfo.isDated ? (
          <PlanSummaryDatingDated
            className=""
            datingInfo={planParamsNorm.datingInfo}
          />
        ) : (
          <PlanSummaryDatingDateless
            className=""
            datingInfo={planParamsNorm.datingInfo}
          />
        )}
      </div>
    )
  },
)
