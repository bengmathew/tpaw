import {
  NonPlanParams,
  assert,
  block,
  getDefaultNonPlanParams,
} from '@tpaw/common'
import clix from 'clsx'
import _ from 'lodash'
import React, { useMemo } from 'react'
import { paddingCSS } from '../../../../../Utils/Geometry'
import { NumberInput } from '../../../../Common/Inputs/NumberInput'
import { smartDeltaFnForMonthlyAmountInput } from '../../../../Common/Inputs/SmartDeltaFnForAmountInput'
import { SwitchAsToggle } from '../../../../Common/Inputs/SwitchAsToggle'
import { useNonPlanParams } from '../../../PlanRootHelpers/WithNonPlanParams'
import { usePlanResultsChartType } from '../../PlanResults/UsePlanResultsChartType'
import { useChartData } from '../../WithPlanResultsChartData'
import { PlanInputModifiedBadge } from '../Helpers/PlanInputModifiedBadge'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from '../PlanInputBody/PlanInputBody'
import {
  useSimulationInfo,
  useSimulationResultInfo,
} from '../../../PlanRootHelpers/WithSimulation'
import { normalizePlanParamsInverse } from '../../../../../Simulator/NormalizePlanParams/NormalizePlanParamsInverse'
import { PortfolioBalanceEstimation } from '../../../PlanRootHelpers/PortfolioBalanceEstimation'
import { appPaths } from '../../../../../AppPaths'

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
    const { planParamsNormInstant } = useSimulationInfo()
    const { simulationResult } = useSimulationResultInfo()
    const isModified = useIsPlanInputDevMiscModified()
    const defaultNonPlanParams = useMemo(
      () => getDefaultNonPlanParams(Date.now()),
      [],
    )

    return (
      <div
        className={`${className} params-card relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
        <div className=" flex justify-start gap-x-4 items-center mt-4">
          <SwitchAsToggle
            className=""
            checked={nonPlanParams.dev.alwaysShowAllMonths}
            setChecked={(x) => {
              const clone = _.cloneDeep(nonPlanParams)
              clone.dev.alwaysShowAllMonths = x
              setNonPlanParams(clone)
            }}
          />
          <h2 className="">Always Show All Months</h2>
        </div>
        <div className=" flex justify-start gap-x-4 items-center mt-4">
          <SwitchAsToggle
            className=""
            checked={nonPlanParams.dev.showSyncStatus}
            setChecked={(x) => {
              const clone = _.cloneDeep(nonPlanParams)
              clone.dev.showSyncStatus = x
              setNonPlanParams(clone)
            }}
          />
          <h2 className=""> Show Sync Status</h2>
        </div>
        <_ChartYRangeOverride className="mt-4" />
        <button
          className="py-2 underline mt-2"
          onClick={() => {
            const params = block(() => {
              const clone = normalizePlanParamsInverse(
                simulationResult.planParamsNormOfResult,
              )
              if (
                clone.wealth.portfolioBalance.isDatedPlan &&
                !clone.wealth.portfolioBalance.updatedHere
              ) {
                assert(
                  simulationResult.planParamsNormOfResult.datingInfo.isDated,
                )
                clone.timestamp =
                  simulationResult.planParamsNormOfResult.datingInfo.nowAsTimestamp
                clone.wealth.portfolioBalance = {
                  isDatedPlan: true,
                  updatedHere: true,
                  amount:
                    simulationResult.portfolioBalanceEstimationByDated
                      .currentBalance,
                }
              }
              return clone
            })
            const url = appPaths.link()
            url.searchParams.set('params', JSON.stringify(params))
            void window.navigator.clipboard.writeText(url.href)
          }}
        >
          Create a Long Link
        </button>

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
          onClick={() => {
            const clone = _.cloneDeep(nonPlanParams)
            clone.dev.showSyncStatus = defaultNonPlanParams.dev.showSyncStatus
            clone.dev.alwaysShowAllMonths =
              defaultNonPlanParams.dev.alwaysShowAllMonths
            clone.dev.overridePlanResultChartYRange =
              defaultNonPlanParams.dev.overridePlanResultChartYRange
            assert(!_getIsPlanInputDevMiscModified(clone))
            setNonPlanParams(clone)
          }}
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
      <div className={clix(className)}>
        <div className="flex justify-start gap-x-4 items-center ">
          <SwitchAsToggle
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
  return _getIsPlanInputDevMiscModified(nonPlanParams)
}
const _getIsPlanInputDevMiscModified = (nonPlanParams: NonPlanParams) => {
  const defaultNonPlanParams = getDefaultNonPlanParams(Date.now())
  return (
    nonPlanParams.dev.showSyncStatus !==
      defaultNonPlanParams.dev.showSyncStatus ||
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
        Always Show All Months:{' '}
        {nonPlanParams.dev.alwaysShowAllMonths ? 'yes' : 'no'}
      </h2>
      <h2>
        Show Sync Status: {nonPlanParams.dev.showSyncStatus ? 'yes' : 'no'}
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
