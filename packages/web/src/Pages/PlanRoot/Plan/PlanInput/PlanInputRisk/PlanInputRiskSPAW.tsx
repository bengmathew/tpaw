import { PLAN_PARAMS_CONSTANTS, partialDefaultDatelessPlanParams } from '@tpaw/common'
import React from 'react'
import { PlanParamsNormalized } from '@tpaw/common'
import { formatPercentage } from '../../../../../Utils/FormatPercentage'
import { paddingCSSStyle } from '../../../../../Utils/Geometry'
import { SliderInput } from '../../../../Common/Inputs/SliderInput/SliderInput'
import { useSimulationInfo } from '../../../PlanRootHelpers/WithSimulation'
import { PlanInputSummaryGlidePath } from '../Helpers/PlanInputSummaryGlidePath'
import { PlanInputBodyPassThruProps } from '../PlanInputBody/PlanInputBody'
import { PlanInputRiskSPAWAndSWRStockAllocationCard } from './PlanInputRiskSPAWAndSWRStockAllocation'

export const PlanInputRiskSPAW = React.memo(
  ({ props }: { props: PlanInputBodyPassThruProps }) => {
    return (
      <>
        <PlanInputRiskSPAWAndSWRStockAllocationCard
          className=""
          props={props}
        />
        <_SpendingTiltCard className="mt-8" props={props} />
      </>
    )
  },
)

const _SpendingTiltCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { planParamsNormInstant, updatePlanParams } = useSimulationInfo()
    const handleChange = (value: number) =>
      updatePlanParams('setSPAWAnnualSpendingTilt', value)

    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg">Spending Tilt</h2>
        <p className="p-base mt-2">
          This lets you shift your spending between early and late retirement.
          To spend more in early retirement and less in late retirement, move
          the slider to the left. To spend more in late retirement and less in
          early retirement, move the slider to the right.
        </p>
        <SliderInput
          className="-mx-3 mt-2"
          height={60}
          maxOverflowHorz={props.sizing.cardPadding}
          data={PLAN_PARAMS_CONSTANTS.risk.spaw.annualSpendingTilt.values}
          value={planParamsNormInstant.risk.spaw.annualSpendingTilt}
          onChange={handleChange}
          format={formatPercentage(1)}
          ticks={(value, i) =>
            i % 10 === 0 ? 'large' : i % 1 === 0 ? 'small' : 'none'
          }
        />
        <button
          className="mt-4 underline"
          onClick={() =>
            handleChange(partialDefaultDatelessPlanParams.risk.spaw.annualSpendingTilt)
          }
        >
          Reset to Default
        </button>
      </div>
    )
  },
)

export const PlanInputRiskSPAWSummary = React.memo(
  ({ planParamsNormInstant }: { planParamsNormInstant: PlanParamsNormalized }) => {
    const { risk } = planParamsNormInstant
    return (
      <>
        <h2>Stock Allocation</h2>
        <div className="ml-4">
          <PlanInputSummaryGlidePath
            className=""
            normValue={risk.spawAndSWR.allocation}
          />
        </div>
        <h2>
          Spending Tilt: {formatPercentage(1)(risk.spaw.annualSpendingTilt)}
        </h2>
      </>
    )
  },
)
