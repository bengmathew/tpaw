import { ADDITIONAL_ANNUAL_SPENDING_TILT_VALUES } from '@tpaw/common'
import React from 'react'
import { formatPercentage } from '../../../../../Utils/FormatPercentage'
import { paddingCSS } from '../../../../../Utils/Geometry'
import { SliderInput } from '../../../../Common/Inputs/SliderInput/SliderInput'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'
import { PlanInputModifiedBadge } from '../Helpers/PlanInputModifiedBadge'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from '../PlanInputBody/PlanInputBody'

export const PlanInputDevAdditionalSpendingTilt = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    return (
      <PlanInputBody {...props}>
        <div className="">
          <_AdditionalSpendingTiltCard className="mt-10" props={props} />
        </div>
      </PlanInputBody>
    )
  },
)
const _AdditionalSpendingTiltCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { planParams, updatePlanParams, defaultPlanParams } = useSimulation()
    const isModified = useIsPlanInputDevAdditionalSpendingTiltModified()

    const handleChange = (value: number) =>
      updatePlanParams('setTPAWAdditionalSpendingTilt', value)
    return (
      <div
        className={`${className} params-card relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
        <p className="p-base mt-2">
          This lets you shift your spending between early and late retirement.
          To spend more in early retirement and less in late retirement, move
          the slider to the left. To spend more in late retirement and less in
          early retirement, move the slider to the right.
        </p>

        <SliderInput
          className={`-mx-3 mt-2 `}
          height={60}
          maxOverflowHorz={props.sizing.cardPadding}
          data={ADDITIONAL_ANNUAL_SPENDING_TILT_VALUES}
          value={planParams.risk.tpaw.additionalAnnualSpendingTilt}
          onChange={(x) => handleChange(x)}
          format={(x) => formatPercentage(1)(x)}
          ticks={(value, i) => (i % 10 === 0 ? 'large' : 'small')}
        />
        <button
          className="mt-6 underline disabled:lighten-2"
          onClick={() =>
            handleChange(
              defaultPlanParams.risk.tpaw.additionalAnnualSpendingTilt,
            )
          }
          disabled={!isModified}
        >
          Reset to Default
        </button>
      </div>
    )
  },
)

export const useIsPlanInputDevAdditionalSpendingTiltModified = () => {
  const { planParams, defaultPlanParams } = useSimulation()
  return (
    defaultPlanParams.risk.tpaw.additionalAnnualSpendingTilt !==
    planParams.risk.tpaw.additionalAnnualSpendingTilt
  )
}
export const PlanInputDevAdditionalSpendingTiltSummary = React.memo(() => {
  const { planParams } = useSimulation()
  return (
    <h2>
      {formatPercentage(1)(planParams.risk.tpaw.additionalAnnualSpendingTilt)}
    </h2>
  )
})
