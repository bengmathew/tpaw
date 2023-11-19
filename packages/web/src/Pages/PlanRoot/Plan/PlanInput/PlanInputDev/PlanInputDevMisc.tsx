import { defaultNonPlanParams } from '@tpaw/common'
import clsx from 'clsx'
import _ from 'lodash'
import React from 'react'
import { paddingCSS } from '../../../../../Utils/Geometry'
import { NumberInput } from '../../../../Common/Inputs/NumberInput'
import { smartDeltaFnForMonthlyAmountInput } from '../../../../Common/Inputs/SmartDeltaFnForAmountInput'
import { ToggleSwitch } from '../../../../Common/Inputs/ToggleSwitch'
import { useNonPlanParams } from '../../../PlanRootHelpers/WithNonPlanParams'
import { useGetPlanResultsChartURL } from '../../PlanResults/UseGetPlanResultsChartURL'
import { usePlanResultsChartType } from '../../PlanResults/UsePlanResultsChartType'
import { useChartData } from '../../WithPlanResultsChartData'
import { PlanInputModifiedBadge } from '../Helpers/PlanInputModifiedBadge'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from '../PlanInputBody/PlanInputBody'

export const PlanInputDevMisc = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    return (
      <PlanInputBody {...props}>
        <div className="">
          <_MiscCard className="mt-10" props={props} />
        </div>
      </PlanInputBody>
    )
  },
)

const _MiscCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { nonPlanParams, setNonPlanParams } = useNonPlanParams()
    const getPlanChartURL = useGetPlanResultsChartURL()

    const isModified = useIsPlanInputDevMiscModified()

    const handleChangeShowAllMonths = (x: boolean) => {
      const clone = _.cloneDeep(nonPlanParams)
      clone.dev.alwaysShowAllMonths = x
      setNonPlanParams(clone)
    }

    return (
      <div
        className={`${className} params-card relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
        <div className=" flex justify-start gap-x-4 items-center mt-4">
          <ToggleSwitch
            className=""
            checked={nonPlanParams.dev.alwaysShowAllMonths}
            setChecked={(x) => handleChangeShowAllMonths(x)}
          />
          <h2 className=""> Show All Months</h2>
        </div>
        <_ChartYRangeOverride className="mt-4" />

        <button className="block btn-sm btn-outline mt-4" onClick={() => {}}>
          Test
        </button>
        <button
          className="block btn-sm btn-outline mt-4"
          onClick={() => {
            throw new Error('Crash Test')
          }}
        >
          Crash
        </button>
        <button
          className="mt-6 underline disabled:lighten-2 block"
          onClick={() =>
            handleChangeShowAllMonths(
              defaultNonPlanParams.dev.alwaysShowAllMonths,
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

const _ChartYRangeOverride = React.memo(
  ({ className }: { className?: string }) => {
    const { nonPlanParams, setNonPlanParams } = useNonPlanParams()
    const chartType = usePlanResultsChartType()
    const chartData = useChartData(chartType)
    return (
      <div className={clsx(className)}>
        <div className="flex justify-start gap-x-4 items-center ">
          <ToggleSwitch
            checked={!!nonPlanParams.dev.overridePlanResultChartYRange}
            setChecked={(checked) => {
              const clone = _.cloneDeep(nonPlanParams)
              if (!checked) {
                clone.dev.overridePlanResultChartYRange = false
              } else {
                clone.dev.overridePlanResultChartYRange =
                  chartData.displayRange.y
              }
              setNonPlanParams(clone)
            }}
          />
          <h2 className=""> Override Y Range</h2>
        </div>
        {!!nonPlanParams.dev.overridePlanResultChartYRange && (
          <div
            className="ml-[50px] mt-3 inline-grid gap-x-4 items-center"
            style={{ grid: 'auto/auto 100px' }}
          >
            <h2 className="">Max Y</h2>
            <NumberInput
              value={nonPlanParams.dev.overridePlanResultChartYRange.end}
              textAlign="right"
              width={125}
              setValue={(end) => {
                if (end <= 0) return true
                const clone = _.cloneDeep(nonPlanParams)
                clone.dev.overridePlanResultChartYRange = { start: 0, end }
                setNonPlanParams(clone)
                return false
              }}
              modalLabel={'Max Y'}
              increment={smartDeltaFnForMonthlyAmountInput.increment}
              decrement={smartDeltaFnForMonthlyAmountInput.decrement}
            />
          </div>
        )}
      </div>
    )
  },
)

export const useIsPlanInputDevMiscModified = () => {
  const { nonPlanParams } = useNonPlanParams()
  return (
    nonPlanParams.dev.alwaysShowAllMonths !==
      defaultNonPlanParams.dev.alwaysShowAllMonths ||
    nonPlanParams.dev.overridePlanResultChartYRange !==
      defaultNonPlanParams.dev.overridePlanResultChartYRange
  )
}

export const PlanInputDevMiscSummary = React.memo(() => {
  const { nonPlanParams } = useNonPlanParams()
  return (
    <>
      <h2>
        Always show all months:{' '}
        {nonPlanParams.dev.alwaysShowAllMonths ? 'yes' : 'no'}
      </h2>
      <h2 className="">
        Override Y Range:{' '}
        {nonPlanParams.dev.overridePlanResultChartYRange
          ? `${new Intl.NumberFormat('en-US', {
              minimumFractionDigits: 0,
              maximumFractionDigits: 0,
            }).format(nonPlanParams.dev.overridePlanResultChartYRange.end)}`
          : 'no'}
      </h2>
    </>
  )
})
