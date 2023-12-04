import { faExclamation } from '@fortawesome/pro-solid-svg-icons'
import { faPlus } from '@fortawesome/pro-thin-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import clix from 'clsx'
import React from 'react'
import { Padding, paddingCSSStyle } from '../../../../Utils/Geometry'
import { useURLUpdater } from '../../../../Utils/UseURLUpdater'
import { noCase } from '../../../../Utils/Utils'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
import { useGetSectionURL } from '../Plan'
import { PlanInputModifiedBadge } from '../PlanInput/Helpers/PlanInputModifiedBadge'
import { PlanInputType } from '../PlanInput/Helpers/PlanInputType'
import { planSectionLabel } from '../PlanInput/Helpers/PlanSectionLabel'
import { PlanSectionName } from '../PlanInput/Helpers/PlanSectionName'
import { useGetPlanInputVisibility } from '../PlanInput/Helpers/UseGetPlanInputVisibility'
import { PlanInputAgeSummary } from '../PlanInput/PlanInputAge/PlanInputAge'
import { PlanInputCurrentPortfolioBalanceSummary } from '../PlanInput/PlanInputCurrentPortfolioBalance'
import { PlanInputDevHistoricalReturnsSummary } from '../PlanInput/PlanInputDev/PlanInputDevHistoricalReturns'
import { PlanInputDevMiscSummary } from '../PlanInput/PlanInputDev/PlanInputDevMisc'
import { PlanInputDevSimulationsSummary } from '../PlanInput/PlanInputDev/PlanInputDevSimulations'
import { PlanInputDevTimeSummary } from '../PlanInput/PlanInputDev/PlanInputDevTime'
import { PlanInputExpectedReturnsAndVolatilitySummary } from '../PlanInput/PlanInputExpectedReturnsAndVolatility'
import { PlanInputExtraSpendingSummary } from '../PlanInput/PlanInputExtraSpending'
import { PlanInputFutureSavingsSummary } from '../PlanInput/PlanInputFutureSavings'
import { PlanInputIncomeDuringRetirementSummary } from '../PlanInput/PlanInputIncomeDuringRetirement'
import { PlanInputInflationSummary } from '../PlanInput/PlanInputInflation'
import { PlanInputLegacySummary } from '../PlanInput/PlanInputLegacy'
import { PlanInputRiskSummary } from '../PlanInput/PlanInputRisk/PlanInputRisk'
import { PlanInputSimulationSummary } from '../PlanInput/PlanInputSimulation'
import { PlanInputSpendingCeilingAndFloorSummary } from '../PlanInput/PlanInputSpendingCeilingAndFloor'
import { PlanInputStrategySummary } from '../PlanInput/PlanInputStrategy'
import { gray, orange } from '../../../../Utils/ColorPalette'
import { usePlanColors } from '../UsePlanColors'

type _Props = {
  padding: Padding
  type: Exclude<PlanInputType, 'history'>
  section: PlanSectionName
  hideTitle?: boolean
  warn?: boolean
  flagAsModified?: boolean
  empty?: boolean
}
export const PlanSummaryButton = React.memo(
  React.forwardRef<HTMLButtonElement, _Props>(
    (
      {
        padding,
        type,
        section,
        warn = false,
        flagAsModified = false,
        empty = false,
        hideTitle = false,
      }: _Props,
      ref,
    ) => {
      const { planParams, planParamsExt } = useSimulation()
      const { dialogPositionEffective } = planParamsExt
      const getSectionURL = useGetSectionURL()
      const urlUpdater = useURLUpdater()
      const highlightColorDark = gray[400]
      const visibility = useGetPlanInputVisibility()(type)
      const highlightColor =
        section === type
          ? highlightColorDark
          : dialogPositionEffective === type
          ? orange[400]
          : gray[100]

      const planColors = usePlanColors()

      if (!visibility.visible) return <></>

      return (
        <button
          className={clix(
            'block rounded-2xl  text-left w-full border-[2px] overflow-hidden',
            visibility.grayOutButton && 'opacity-20',
            planColors.summaryButtonOuter(dialogPositionEffective === type),
          )}
          ref={ref}
          style={{
            transitionProperty: 'border-color',
            transitionDuration:
              highlightColor === highlightColorDark ? '500ms' : '1250ms',
            borderColor: highlightColor,
          }}
          onClick={() => urlUpdater.push(getSectionURL(type))}
          disabled={visibility.disabled}
        >
          {empty ? (
            <div
              className={`relative  bg-gray-200/40  rounded-2xl -m-[2px] 
              ${
                dialogPositionEffective === type
                  ? 'border-[3px] border-gray-200'
                  : 'border-[4px] border-dotted border-gray-400'
              }`}
              style={{ ...paddingCSSStyle(padding) }}
            >
              <div className=" flex items-center gap-x-2 mb-1">
                <FontAwesomeIcon className="text-3xl" icon={faPlus} />
                <h2 className="font-semibold mr-2 opacity-[0.75] ">
                  <span className="">Add {planSectionLabel(type)}</span>
                </h2>
              </div>
            </div>
          ) : (
            <div
              className="relative border-[3px] border-gray-200 rounded-2xl -m-[2px]"
              style={{ ...paddingCSSStyle(padding) }}
            >
              <PlanInputModifiedBadge show={flagAsModified} mainPage />
              {(!hideTitle || warn) && (
                <div className="flex  items-center mb-1">
                  {!hideTitle && (
                    <h2 className="font-semibold mr-2 flex">
                      <span className="">{planSectionLabel(type)}</span>
                    </h2>
                  )}
                  {warn && (
                    <div className="h-[18px] w-[18px] flex items-center justify-center text-[13px] font-bold rounded-full bg-errorBlockBG text-errorBlockFG">
                      <FontAwesomeIcon icon={faExclamation} />
                    </div>
                  )}
                </div>
              )}
              <div className="grid gap-y-1 text-sm text-gray-600">
                <_SectionSummary type={type} />
              </div>
            </div>
          )}
        </button>
      )
    },
  ),
)

const _SectionSummary = React.memo(
  ({ type }: { type: Exclude<PlanInputType, 'history'> }) => {
    switch (type) {
      case 'age':
        return <PlanInputAgeSummary />
      case 'current-portfolio-balance':
        return <PlanInputCurrentPortfolioBalanceSummary />
      case 'future-savings':
        return <PlanInputFutureSavingsSummary />
      case 'income-during-retirement':
        return <PlanInputIncomeDuringRetirementSummary />
      case 'extra-spending':
        return <PlanInputExtraSpendingSummary />
      case 'legacy':
        return <PlanInputLegacySummary />
      case 'spending-ceiling-and-floor':
        return <PlanInputSpendingCeilingAndFloorSummary />
      case 'risk':
        return <PlanInputRiskSummary />
      case 'expected-returns-and-volatility':
        return <PlanInputExpectedReturnsAndVolatilitySummary />
      case 'inflation':
        return <PlanInputInflationSummary />
      case 'strategy':
        return <PlanInputStrategySummary />
      case 'simulation':
        return <PlanInputSimulationSummary />
      case 'dev-misc':
        return <PlanInputDevMiscSummary />
      case 'dev-simulations':
        return <PlanInputDevSimulationsSummary />
      case 'dev-historical-returns':
        return <PlanInputDevHistoricalReturnsSummary />
      case 'dev-time':
        return <PlanInputDevTimeSummary />
      default:
        noCase(type)
    }
  },
)
