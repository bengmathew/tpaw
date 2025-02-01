import {
  faArrowLeftLong,
  faArrowRightLong,
  faCaretDown,
  faCaretRight,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  DEFAULT_RISK_TPAW,
  PLAN_PARAMS_CONSTANTS,
  block,
  fGet,
  monthlyToAnnualReturnRate,
  partialDefaultDatelessPlanParams
} from '@tpaw/common'
import clix from 'clsx'
import _ from 'lodash'
import React, { ReactNode, useState } from 'react'
import { PlanParamsNormalized } from '../../../../../Simulator/NormalizePlanParams/NormalizePlanParams'
import { SimulationResult } from '../../../../../Simulator/UseSimulator'
import { formatPercentage } from '../../../../../Utils/FormatPercentage'
import { paddingCSS, paddingCSSStyle } from '../../../../../Utils/Geometry'
import { SliderInput } from '../../../../Common/Inputs/SliderInput/SliderInput'
import {
  useSimulationInfo,
  useSimulationResultInfo,
} from '../../../PlanRootHelpers/WithSimulation'
import { PlanInputModifiedBadge } from '../Helpers/PlanInputModifiedBadge'
import { PlanInputBodyPassThruProps } from '../PlanInputBody/PlanInputBody'
import { PlanInputRiskRRASlider } from './PlanInputRiskRRASlider'

const RISK_TOLERANCE_VALUES = _.range(
  0,
  PLAN_PARAMS_CONSTANTS.risk.tpaw.riskTolerance.values
    .numIntegerValuesStartingFrom0,
)
export const PlanInputRiskTPAW = React.memo(
  ({ props }: { props: PlanInputBodyPassThruProps }) => {
    const advancedCount = _.filter([
      useIsRiskToleranceDeclineCardModified(),
      useIsLegacyRiskToleranceDeltaCardModified(),
      useIsTimePreferenceCardModified(),
    ]).length

    const [showAdvanced, setShowAdvanced] = useState(false)
    return (
      <div className="">
        <_TPAWRiskToleranceCard className="" props={props} />
        <_SpendingTiltCard className="mt-10" props={props} />
        <button
          className="mt-6 text-start"
          onClick={() => setShowAdvanced((x) => !x)}
          style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
        >
          <div className="text-[20px] sm:text-[24px] font-bold text-left">
            Advanced{' '}
            <FontAwesomeIcon icon={showAdvanced ? faCaretDown : faCaretRight} />
          </div>
          <h2 className="">
            {advancedCount === 0
              ? 'None modified'
              : `${advancedCount} modified`}
          </h2>
        </button>
        {showAdvanced && (
          <>
            <_TPAWRiskToleranceDeclineCard className="" props={props} />
            <_TPAWLegacyRiskToleranceDeltaCard
              className="mt-10"
              props={props}
            />
            <_TPAWTimePreferenceCard className="mt-10" props={props} />
          </>
        )}
      </div>
    )
  },
)

const _TPAWRiskToleranceCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { planParamsNormInstant, updatePlanParams } = useSimulationInfo()
    const { simulationResult } = useSimulationResultInfo()
    const defaultRisk = partialDefaultDatelessPlanParams.risk.tpaw

    const stockAllocations = block(() => {
      const get50thStockAllocation = (mfn: number) =>
        fGet(
          simulationResult.savingsPortfolio.afterWithdrawals.allocation.stocks.byPercentileByMonthsFromNow.find(
            (x) => x.percentile === 50,
          ),
        ).data[mfn]

      const effectiveMaxAgeAsMFN =
        simulationResult.planParamsNormOfResult.ages.simulationMonths
          .lastMonthAsMFN +
        (simulationResult.planParamsNormOfResult.adjustmentsToSpending
          .tpawAndSPAW.legacy.total > 0
          ? 0
          : -1)
      return {
        now: get50thStockAllocation(0),
        atRetirement: get50thStockAllocation(
          simulationResult.planParamsNormOfResult.ages.simulationMonths
            .withdrawalStartMonth.asMFN,
        ),
        atMaxAge: get50thStockAllocation(effectiveMaxAgeAsMFN),
      }
    })
    return (
      <div
        className={`${className} params-card`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <h2 className="font-bold text-lg mb-2">Risk Tolerance</h2>
        <p className="p-base">
          How much risk do you want to take on your retirement spending? More
          risk leads to higher average spending, but also a wider range of
          outcomes.
        </p>
        <div className="mt-8 bg-gray-100  rounded-lg border border-gray-200 py-4">
          <div className="flex justify-between mx-[15px]">
            <div className="flex items-center gap-x-2">
              <FontAwesomeIcon icon={faArrowLeftLong} />
              Conservative
            </div>
            <div className="flex items-center gap-x-2">
              Aggressive
              <FontAwesomeIcon icon={faArrowRightLong} />
            </div>
          </div>
          <PlanInputRiskRRASlider
            className={` `}
            height={60}
            maxOverflowHorz={props.sizing.cardPadding}
            data={RISK_TOLERANCE_VALUES}
            value={planParamsNormInstant.risk.tpaw.riskTolerance.at20}
            onChange={(value) =>
              updatePlanParams('setTPAWRiskTolerance', value)
            }
            format={(value) => `${value}`}
            ticks={() => 'small'}
          />
        </div>
        <_ExpandableNote
          className="mt-8"
          title="Stock allocation corresponding to this risk tolerance"
        >
          <p className="p-base">
            Your risk tolerance, together with other inputs such as your essential
            expenses and retirement income and the equity premium, determine how
            much of your portfolio should be allocated to stocks versus bonds.
            Your current inputs result in the following stock allocation:
          </p>
          <h2 className="font-semibold mt-4">Stock Allocation</h2>
          <div
            className="grid gap-x-4  mb-4 mt-2"
            style={{ grid: 'auto/auto auto 1fr ' }}
          >
            <h2>Now</h2>
            <h2 className="text-right">
              {formatPercentage(0)(stockAllocations.now)}
            </h2>
            <h2></h2>
            {planParamsNormInstant.ages.simulationMonths.withdrawalStartMonth
              .asMFN > 0 && (
              <>
                <h2>At retirement</h2>
                <h2 className="text-right">
                  {formatPercentage(0)(stockAllocations.atRetirement)}
                </h2>
                <h2>(50th percentile)</h2>
              </>
            )}
            <h2>At max age</h2>
            <h2 className="text-right">
              {formatPercentage(0)(stockAllocations.atMaxAge)}
            </h2>
            <h2 className="">(50th percentile)</h2>
          </div>
          <h2 className="font-semibold">
            Why the percentile for future years?
          </h2>
          {/* <p className="p-base mb-4">
            You are seeing percentiles for the stock allocation for future years
            because the optimal allocation will depend on the relative sizes of
            competing spending goals and other resources like pensions. This
            depends on market performance and so we can get a range of possible
            allocations from the simulations.
          </p> */}
          <p className="p-base mt-2">
            Your stock allocation in the future may be a range and not a single
            number because the optimal allocation will depend on the relative
            sizes of competing spending goals and other resources like pensions.
            This depends on market performance and so we can get a range of
            possible allocations from the simulations.
          </p>
          {stockAllocations &&
            stockAllocations.atMaxAge > stockAllocations.atRetirement && (
              <>
                <h2 className="font-semibold mt-4">
                  Why does the stock allocation increase between retirement and
                  max age?
                </h2>
                <p className="p-base mt-2">
                  This happens when you have a legacy goal. As you get older,
                  more of your assets are going towards legacy and less towards
                  funding your remaining retirement years. Since you have a
                  higher risk tolerance for legacy, your portfolio becomes
                  correspondingly more aggressive. You can change your risk
                  tolerance for legacy in advanced settings.
                </p>
              </>
            )}
        </_ExpandableNote>
        <_ExpandableNote
          className="mt-2"
          title="Relative risk aversion (RRA) corresponding to this risk tolerance"
        >
          <p className="p-base">
            Your risk tolerance of{' '}
            {
              simulationResult.planParamsNormOfResult.risk.tpaw.riskTolerance
                .at20
            }{' '}
            corresponds to a relative risk aversion of{' '}
            <span className="font-bold">
              {_rraIncludingPosInfinityToStr(
                simulationResult.planParamsProcessed.risk.tpaw
                  .rraUnclampedAt20IncludingPosInfinity,
              )}
            </span>
            .
          </p>
        </_ExpandableNote>

        <button
          className="mt-6 underline disabled:lighten-2"
          onClick={() =>
            updatePlanParams(
              'setTPAWRiskTolerance',
              defaultRisk.riskTolerance.at20,
            )
          }
          disabled={
            defaultRisk.riskTolerance.at20 ===
            planParamsNormInstant.risk.tpaw.riskTolerance.at20
          }
        >
          Reset to Default
        </button>
      </div>
    )
  },
)

const _ExpandableNote = React.memo(
  ({
    className = '',
    title,
    children,
  }: {
    className?: string
    title: string
    children: ReactNode
  }) => {
    const [open, setOpen] = useState(false)
    const titleSplit = title.split(' ')
    const last = titleSplit.pop()
    const titleFirst = titleSplit.join(' ')
    return (
      <div className={clix(className)}>
        <button className="text-start mb-2" onClick={() => setOpen((x) => !x)}>
          <span className="font-medium ">{titleFirst}</span>{' '}
          {/* So carret is always with at least on word */}
          <span className="font-medium whitespace-nowrap">
            {last}
            <FontAwesomeIcon
              className="ml-2"
              icon={open ? faCaretDown : faCaretRight}
            />
          </span>
        </button>
        {open && (
          <div className="bg-orange-50 border border-gray-300 rounded-lg p-2 sm:p-4 mb-8">
            {children}
          </div>
        )}
      </div>
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
    const { simulationResult } = useSimulationResultInfo()

    const handleChange = (value: number) =>
      updatePlanParams('setTPAWAdditionalSpendingTilt', value)
    const isModified =
      partialDefaultDatelessPlanParams.risk.tpaw
        .additionalAnnualSpendingTilt !==
      planParamsNormInstant.risk.tpaw.additionalAnnualSpendingTilt

    const getSpendingTiltAtMFN = (
      mfn: number,
      { planParamsNormOfResult, tpawSpendingTilt }: SimulationResult,
    ) => {
      const total = monthlyToAnnualReturnRate(
        fGet(
          tpawSpendingTilt.total.byPercentileByMonthsFromNow.find(
            (x) => x.percentile === 50,
          ),
        ).data[mfn],
      )
      const extra =
        planParamsNormOfResult.risk.tpaw.additionalAnnualSpendingTilt
      const baseline = total - extra
      return (
        <>
          <h2 className="text-right  ">{formatPercentage(2)(baseline)}</h2>
          <h2 className="">+</h2>
          <h2 className="text-right  ">{formatPercentage(2)(extra)}</h2>
          <h2 className="">=</h2>
          <h2 className="text-right  ">{formatPercentage(2)(total)}</h2>
        </>
      )
    }
    return (
      <div
        className={`${className} params-card relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <h2 className="text-lg font-bold"> Spending Tilt</h2>
        <p className="p-base mt-2">
          This lets you shift your spending between early and late retirement.
          To spend more in early retirement, move the slider to the left. To
          spend more in late retirement, move the slider to the right.
        </p>

        <SliderInput
          className={`-mx-3 mt-2 `}
          height={60}
          maxOverflowHorz={props.sizing.cardPadding}
          data={
            PLAN_PARAMS_CONSTANTS.risk.tpaw.additionalAnnualSpendingTilt.values
          }
          value={planParamsNormInstant.risk.tpaw.additionalAnnualSpendingTilt}
          onChange={(x) => handleChange(x)}
          format={(x) => formatPercentage(1)(x)}
          ticks={(value, i) => (i % 10 === 0 ? 'large' : 'small')}
        />
        <_ExpandableNote className="mt-2" title="Spending Tilt Breakdown">
          <p className="p-base">Your spending tilt has two components:</p>
          {/* <h2 className="font-semibold mt-2">Base Spending Tilt</h2> */}
          <h2 className="font-semibold mt-4">1. Base Spending Tilt</h2>
          {/* <ol className=" list-decimal ml-4"> */}
          {/* <li className="p-base mt-2"> */}
          <p className="p-base mt-2">
            This is automatically calculated for you based on your risk
            tolerance and preference for the future. The risk tolerance input is
            located above, and the preference for the future input is located in
            the advanced section below.
          </p>
          <p className="p-base mt-2">
            Your base spending tilt will change with age if your risk tolerance
            changes with age. The default setting decreases risk tolerance by{' '}
            {
              -partialDefaultDatelessPlanParams.risk.tpaw.riskTolerance
                .deltaAtMaxAge
            }{' '}
            between age 20 and max age. You can change this in the advanced
            section below.
          </p>
          {/* </li> */}
          <h2 className="font-semibold mt-4">2. Extra Spending Tilt</h2>
          <p className="p-base mt-2">
            This is the value that you entered in the slider above.
          </p>
          <h2 className="font-bold mt-4">Total Spending Tilt</h2>
          {/* </ol> */}
          {/* <h2 className="font-semibold mt-2">Extra Spending Tilt</h2> */}
          <p className="p-base mt-2">
            Your total spending tilt is obtained by adding your base and extra
            spending tilts. This is a table of your total spending tilt at key
            ages:
          </p>
          <div>
            <div
              className="inline-grid mt-4 gap-x-4 rounded-md p-2 border bg-orange-100/30 border-orange-200 items-center"
              style={{ grid: 'auto/auto auto auto auto auto auto' }}
            >
              <h2></h2>
              <h2 className="">Base</h2>
              <h2 className="">+</h2>
              <h2 className="">Extra</h2>
              <h2 className="">=</h2>
              <h2 className="">Total</h2>
              <h2 className="">Now</h2>
              {getSpendingTiltAtMFN(0, simulationResult)}
              {simulationResult.planParamsNormOfResult.ages.simulationMonths
                .withdrawalStartMonth.asMFN > 0 && (
                <>
                  <h2 className="">
                    <div className="">At retirement</div>{' '}
                    {/* TODO: After duration matching, add this back
                         <div className="text-sm lighten -mt-1">
                          (50th percentile)
                        </div> */}
                  </h2>
                  {getSpendingTiltAtMFN(
                    simulationResult.planParamsNormOfResult.ages
                      .simulationMonths.withdrawalStartMonth.asMFN,
                    simulationResult,
                  )}
                </>
              )}
              <h2 className="">
                <div className="">At max age</div>{' '}
                {/*
                    TODO: After duration matching, add this back
                     <div className="text-sm lighten -mt-1">
                      (50th percentile)
                    </div> */}
              </h2>
              {getSpendingTiltAtMFN(
                simulationResult.planParamsNormOfResult.ages.simulationMonths
                  .numMonths - 1,
                simulationResult,
              )}
            </div>
          </div>
        </_ExpandableNote>

        <button
          className="mt-6 underline disabled:lighten-2"
          onClick={() =>
            handleChange(
              partialDefaultDatelessPlanParams.risk.tpaw
                .additionalAnnualSpendingTilt,
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

const _TPAWRiskToleranceDeclineCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { planParamsNormInstant, updatePlanParams } = useSimulationInfo()
    const { simulationResult } = useSimulationResultInfo()
    const defaultRisk = partialDefaultDatelessPlanParams.risk.tpaw
    const isModified = useIsRiskToleranceDeclineCardModified()

    return (
      <div
        className={`${className} params-card relative`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
        <h2 className="font-bold text-lg mb-2">
          Decrease Risk Tolerance With Age
        </h2>
        <p className="p-base">
          You can decrease your risk tolerance with age here. We will assume
          that the risk tolerance you entered above applies at age 20 and that
          it decreases linearly from there to max age by the amount entered
          below.{' '}
        </p>
        {planParamsNormInstant.ages.person2 && (
          <p className="p-base mt-2">
            {planParamsNormInstant.ages.longerLivedPersonType === 'person1'
              ? `This calculation will be based on the ages of the partner who has the longer remaining lifespan. Based on the ages you have entered, you have the longer remaining lifespan.`
              : `This calculation will be based on the ages of the partner who has the longer remaining lifespan. Based on the ages you have entered, your partner has the longer remaining lifespan.`}
          </p>
        )}
        <SliderInput
          className={`-mx-3 mt-2 `}
          height={60}
          maxOverflowHorz={props.sizing.cardPadding}
          data={RISK_TOLERANCE_VALUES.map((x) => -x)}
          value={planParamsNormInstant.risk.tpaw.riskTolerance.deltaAtMaxAge}
          onChange={(value) =>
            updatePlanParams('setTPAWRiskDeltaAtMaxAge', value)
          }
          format={(x) => (-x).toFixed(0)}
          ticks={(__, i) => (i % 10 === 0 ? 'large' : 'small')}
        />
        <_ExpandableNote className="mt-8" title="Risk Tolerance Table">
          {block(() => {
            const data = _.compact([
              {
                label: 'At Age 20',
                mfn: 'at20' as const,
              },
              {
                label: 'Now',
                mfn: 0,
              },
              simulationResult.planParamsNormOfResult.ages.simulationMonths
                .withdrawalStartMonth.asMFN === 0
                ? null
                : {
                    label: 'Retirement',
                    mfn: simulationResult.planParamsNormOfResult.ages
                      .simulationMonths.withdrawalStartMonth.asMFN,
                  },
              {
                label: 'Max Age',
                mfn: simulationResult.planParamsNormOfResult.ages
                  .simulationMonths.lastMonthAsMFN,
              },
            ])
              .map((x) =>
                x.mfn === 'at20'
                  ? {
                      ...x,
                      riskTolerance:
                        simulationResult.planParamsNormOfResult.risk.tpaw
                          .riskTolerance.at20,
                      rraUnclampedIncludingPosInfinity:
                        simulationResult.planParamsProcessed.risk.tpaw
                          .rraUnclampedAt20IncludingPosInfinity,
                    }
                  : {
                      ...x,
                      riskTolerance:
                        simulationResult.planParamsProcessed.risk.tpaw
                          .riskToleranceByMfn[x.mfn],
                      rraUnclampedIncludingPosInfinity:
                        simulationResult.planParamsProcessed.risk.tpaw
                          .rraUnclampedIncludingPosInfinityByMfn[x.mfn],
                    },
              )

              .sort((a, b) =>
                a.mfn === 'at20' ? -1 : b.mfn === 'at20' ? 1 : a.mfn - b.mfn,
              )
            return (
              <div className="">
                <p className="p-base">
                  This is a table of your risk tolerances at key ages:
                </p>
                <div
                  className="inline-grid max-w-[400px] mt-4 gap-x-4 border border-orange-200 bg-orange-100/30 rounded-md p-2"
                  style={{ grid: 'auto/auto auto auto ' }}
                >
                  <div className="flex items-end justify-center">
                    <h2 className="">
                      {simulationResult.planParamsNormOfResult.ages
                        .longerLivedPersonType === 'person1'
                        ? 'Your Age'
                        : `Your Partner's Age`}
                    </h2>
                  </div>
                  <div className="flex items-end">
                    <h2 className="">Risk Tolerance</h2>
                  </div>
                  <h2 className="text-center">Relative Risk Aversion (RRA)</h2>
                  <h2 className="col-span-3 my-1 -mx-2 border-b border-orange-200"></h2>
                  {data.map(
                    ({
                      label,
                      rraUnclampedIncludingPosInfinity,
                      riskTolerance,
                    }) => (
                      <React.Fragment key={label}>
                        <h2 className="">{label}</h2>
                        <h2 className="text-center">
                          {riskTolerance.toFixed(1)}
                        </h2>
                        <h2 className="text-center">
                          {_rraIncludingPosInfinityToStr(
                            rraUnclampedIncludingPosInfinity,
                          )}
                        </h2>
                      </React.Fragment>
                    ),
                  )}
                </div>
              </div>
            )
          })}
        </_ExpandableNote>

        <button
          className="mt-6 underline disabled:lighten-2"
          onClick={() =>
            updatePlanParams(
              'setTPAWRiskDeltaAtMaxAge',
              defaultRisk.riskTolerance.deltaAtMaxAge,
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

const useIsRiskToleranceDeclineCardModified = () => {
  const { planParamsNormInstant } = useSimulationInfo()
  return (
    partialDefaultDatelessPlanParams.risk.tpaw.riskTolerance.deltaAtMaxAge !==
    planParamsNormInstant.risk.tpaw.riskTolerance.deltaAtMaxAge
  )
}

const _TPAWLegacyRiskToleranceDeltaCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { planParamsNormInstant, updatePlanParams } = useSimulationInfo()
    const defaultRisk = partialDefaultDatelessPlanParams.risk.tpaw
    const isModified = useIsLegacyRiskToleranceDeltaCardModified()

    const handleChange = (value: number) =>
      updatePlanParams('setTPAWRiskToleranceForLegacyAsDeltaFromAt20', value)
    return (
      <div
        className={`${className} params-card relative`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
        <h2 className="font-bold text-lg mb-2">
          Increase Risk Tolerance for Legacy
        </h2>
        <p className="p-base">
          If you have a higher risk tolerance for legacy than for retirement
          spending, you can express it here. Your risk tolerance for legacy is
          obtained by adding the number below to the risk tolerance you entered
          in the main section above.
        </p>
        <SliderInput
          className={`-mx-3 mt-2 `}
          height={60}
          maxOverflowHorz={props.sizing.cardPadding}
          data={RISK_TOLERANCE_VALUES}
          value={
            planParamsNormInstant.risk.tpaw.riskTolerance
              .forLegacyAsDeltaFromAt20
          }
          onChange={handleChange}
          format={(x) => x.toFixed(0)}
          ticks={(value, i) => (i % 10 === 0 ? 'large' : 'small')}
        />
        <button
          className="mt-6 underline disabled:lighten-2"
          onClick={() =>
            handleChange(defaultRisk.riskTolerance.forLegacyAsDeltaFromAt20)
          }
          disabled={!isModified}
        >
          Reset to Default
        </button>
      </div>
    )
  },
)

const useIsLegacyRiskToleranceDeltaCardModified = () => {
  const { planParamsNormInstant } = useSimulationInfo()
  return (
    partialDefaultDatelessPlanParams.risk.tpaw.riskTolerance
      .forLegacyAsDeltaFromAt20 !==
    planParamsNormInstant.risk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20
  )
}

const _TPAWTimePreferenceCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { planParamsNormInstant, updatePlanParams } = useSimulationInfo()
    const defaultRisk = partialDefaultDatelessPlanParams.risk.tpaw
    const isModified = useIsTimePreferenceCardModified()
    const handleChange = (value: number) =>
      updatePlanParams('setTPAWTimePreference', value)
    return (
      <div
        className={`${className} params-card relative`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
        <h2 className="font-bold text-lg mb-2">Preference for the Future</h2>
        <p className="p-base">
          This is a measure of how much you value spending later versus earlier.
          A preference for the future of{' '}
          <span className=" font-font1 text-base italic">x%</span> means that
          next year’s spending is worth{' '}
          <span className=" font-font1 text-base italic">x%</span> more to you
          than this year’s spending.
        </p>
        <SliderInput
          className={`-mx-3 mt-4 `}
          height={60}
          maxOverflowHorz={props.sizing.cardPadding}
          data={[
            ...PLAN_PARAMS_CONSTANTS.risk.tpaw.timePreference.values,
          ].reverse()}
          value={planParamsNormInstant.risk.tpaw.timePreference}
          onChange={(x) => handleChange(x)}
          format={(x) => formatPercentage(1)(-x)}
          ticks={(value, i) => (i % 10 === 0 ? 'large' : 'small')}
        />
        <_ExpandableNote
          className="mt-8"
          title="Time preference rate (TPR) corresponding to this entry"
        >
          <p className="p-base">
            Time preference rate is the negative of your preference for the
            future. So your preference for the future of{' '}
            {formatPercentage(1)(
              -planParamsNormInstant.risk.tpaw.timePreference,
            )}{' '}
            corresponds to a time preference rate of{' '}
            <span className="font-bold">
              {formatPercentage(1)(
                planParamsNormInstant.risk.tpaw.timePreference,
              )}
            </span>
            .
          </p>
        </_ExpandableNote>
        <button
          className="mt-6 underline disabled:lighten-2"
          onClick={() => handleChange(defaultRisk.timePreference)}
          disabled={!isModified}
        >
          Reset to Default
        </button>
      </div>
    )
  },
)
const useIsTimePreferenceCardModified = () => {
  const { planParamsNormInstant } = useSimulationInfo()
  return (
    partialDefaultDatelessPlanParams.risk.tpaw.timePreference !==
    planParamsNormInstant.risk.tpaw.timePreference
  )
}

export const PlanInputRiskTPAWSummary = React.memo(
  ({
    planParamsNormInstant,
  }: {
    planParamsNormInstant: PlanParamsNormalized
  }) => {
    const { risk } = planParamsNormInstant
    const advancedCount = _.filter([
      risk.tpaw.riskTolerance.deltaAtMaxAge !==
        DEFAULT_RISK_TPAW.riskTolerance.deltaAtMaxAge,
      risk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20 !==
        DEFAULT_RISK_TPAW.riskTolerance.forLegacyAsDeltaFromAt20,
      risk.tpaw.timePreference !== DEFAULT_RISK_TPAW.timePreference,
    ]).length
    return (
      <>
        <h2>
          {`Risk Tolerance: ${risk.tpaw.riskTolerance.at20} (${fGet(
            PLAN_PARAMS_CONSTANTS.risk.tpaw.riskTolerance.values.segments.find(
              (x) => x.containsIndex(risk.tpaw.riskTolerance.at20),
            ),
          ).label.toLowerCase()})`}
        </h2>
        <h2>
          Spending Tilt: Base +{' '}
          {formatPercentage(1)(
            planParamsNormInstant.risk.tpaw.additionalAnnualSpendingTilt,
          )}
        </h2>
        {advancedCount > -0 && (
          <>
            <h2 className="">Advanced</h2>
            {DEFAULT_RISK_TPAW.riskTolerance.deltaAtMaxAge !==
              risk.tpaw.riskTolerance.deltaAtMaxAge && (
              <h2 className="ml-4">
                Decrease Risk Tolerance With Age:{' '}
                {-risk.tpaw.riskTolerance.deltaAtMaxAge}
              </h2>
            )}
            {DEFAULT_RISK_TPAW.riskTolerance.forLegacyAsDeltaFromAt20 !==
              risk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20 && (
              <h2 className="ml-4">
                Increase Risk Tolerance For Legacy:{' '}
                {risk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20}
              </h2>
            )}
            {DEFAULT_RISK_TPAW.timePreference !== risk.tpaw.timePreference && (
              <h2 className="ml-4">
                Preference for the Future:{' '}
                {formatPercentage(1)(-risk.tpaw.timePreference)}
              </h2>
            )}
          </>
        )}
      </>
    )
  },
)

const _rraIncludingPosInfinityToStr = (rra: number) =>
  rra === Infinity ? 'infinity' : `${rra.toFixed(2)}`
