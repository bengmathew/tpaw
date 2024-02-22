import { SPAW_ANNUAL_SPENDING_TILT_VALUES } from '@tpaw/common'
import React from 'react'
import { PlanParamsExtended } from '../../../../../UseSimulator/ExtentPlanParams'
import { formatPercentage } from '../../../../../Utils/FormatPercentage'
import { paddingCSSStyle } from '../../../../../Utils/Geometry'
import { SliderInput } from '../../../../Common/Inputs/SliderInput/SliderInput'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'
import { PlanInputSummaryGlidePath } from '../Helpers/PlanInputSummaryGlidePath'
import { PlanInputBodyPassThruProps } from '../PlanInputBody/PlanInputBody'
import { PlanInputRiskLMPCard } from './PlanInputRiskLMPCard'
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
        <PlanInputRiskLMPCard className="mt-8" props={props} />
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
    const { planParams, defaultPlanParams, updatePlanParams } = useSimulation()
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
          data={SPAW_ANNUAL_SPENDING_TILT_VALUES}
          value={planParams.risk.spaw.annualSpendingTilt}
          onChange={handleChange}
          format={formatPercentage(1)}
          ticks={(value, i) =>
            i % 10 === 0 ? 'large' : i % 1 === 0 ? 'small' : 'none'
          }
        />
        <button
          className="mt-4 underline"
          onClick={() =>
            handleChange(defaultPlanParams.risk.spaw.annualSpendingTilt)
          }
        >
          Reset to Default
        </button>
      </div>
    )
  },
)

export const PlanInputRiskSPAWSummary = React.memo(
  ({ planParamsExt }: { planParamsExt: PlanParamsExtended }) => {
    const { planParams } = planParamsExt
    const { risk } = planParams
    return (
      <>
        <h2>Stock Allocation</h2>
        <div className="ml-4">
          <PlanInputSummaryGlidePath
            className=""
            glidePath={risk.spawAndSWR.allocation}
            format={(x) => formatPercentage(0)(x)}
            planParamsExt={planParamsExt}
          />
        </div>
        <h2>
          Spending Tilt: {formatPercentage(1)(risk.spaw.annualSpendingTilt)}
        </h2>
      </>
    )
  },
)
