import {
  faArrowLeftLong,
  faArrowRightLong,
  faCaretDown,
  faCaretRight,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  fGet,
  getDefaultPlanParams,
  RISK_TOLERANCE_VALUES,
  TIME_PREFERENCE_VALUES,
} from '@tpaw/common'
import _ from 'lodash'
import React, { ReactNode, useMemo, useState } from 'react'
import { formatPercentage } from '../../../../Utils/FormatPercentage'
import { paddingCSSStyle } from '../../../../Utils/Geometry'
import { useSimulation } from '../../../App/WithSimulation'
import { SliderInput } from '../../../Common/Inputs/SliderInput/SliderInput'
import { PlanInputModifiedBadge } from '../Helpers/PlanInputModifiedBadge'
import { PlanInputBodyPassThruProps } from '../PlanInputBody/PlanInputBody'
import { PlanInputRiskLMPCard } from './PlanInputRiskLMPCard'
import { PlanInputRiskRRASlider } from './PlanInputRiskRRASlider'

export const PlanInputRiskTPAW = React.memo(
  ({ props }: { props: PlanInputBodyPassThruProps }) => {
    const defaultRisk = useMemo(() => getDefaultPlanParams().risk.tpaw, [])
    const risk = useSimulation().params.risk.tpaw
    const advancedCount = _.filter([
      risk.riskTolerance.deltaAtMaxAge !==
        defaultRisk.riskTolerance.deltaAtMaxAge,
      risk.riskTolerance.forLegacyAsDeltaFromAt20 !==
        defaultRisk.riskTolerance.forLegacyAsDeltaFromAt20,
      risk.timePreference !== defaultRisk.timePreference,
    ]).length

    const [showAdvanced, setShowAdvanced] = useState(false)
    return (
      <div className="">
        <_TPAWRelativeRiskToleranceCard className="" props={props} />
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
            {advancedCount === 0 ? 'None' : `${advancedCount} modified`}
          </h2>
        </button>
        {showAdvanced && (
          <>
            <_TPAWRiskToleranceDeclineCard className="" props={props} />
            <_TPAWLegacyRiskToleranceDeltaCard className="mt-8" props={props} />
            <_TPAWSpendingTiltCard className="mt-8" props={props} />
            <PlanInputRiskLMPCard className="mt-8" props={props} />
          </>
        )}
      </div>
    )
  },
)

const _TPAWRelativeRiskToleranceCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { params, paramsExt, setParams, tpawResult } = useSimulation()
    const defaultRisk = getDefaultPlanParams().risk.tpaw
    const { asYFN, withdrawalStartYear, withdrawalsStarted, maxMaxAge } =
      paramsExt

    const handleChange = (value: number) =>
      setParams((params) => {
        const clone = _.cloneDeep(params)
        clone.risk.tpaw.riskTolerance.at20 = value
        return clone
      })

    const get50thStockAllocation = (yfn: number) =>
      fGet(
        tpawResult.savingsPortfolio.afterWithdrawals.allocation.stocks.byPercentileByYearsFromNow.find(
          (x) => x.percentile === 50,
        ),
      ).data[yfn]

    const effectiveMaxAgeAYFN =
      asYFN(maxMaxAge) +
      (params.adjustmentsToSpending.tpawAndSPAW.legacy.total > 0 ? 0 : -1)
    const stockAllocations = {
      now: get50thStockAllocation(0),
      atRetirement: get50thStockAllocation(asYFN(withdrawalStartYear)),
      atMaxAge: get50thStockAllocation(effectiveMaxAgeAYFN),
    }

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
        <div className="flex justify-between mx-[15px] mt-8">
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
          data={RISK_TOLERANCE_VALUES.DATA}
          value={params.risk.tpaw.riskTolerance.at20}
          onChange={handleChange}
          format={(value) => `${value}`}
          ticks={() => 'small'}
        />
        <div className="mt-2">
          <div
            className="mt-8"
            // title="What stock allocation does this risk tolerance imply?"
          >
            <p className="p-base">
              Your risk tolerance, together with other inputs such as essential
              expenses and retirement income, determine how much of your
              portfolio should be allocated to stocks versus bonds. Your current
              inputs result in the following stock allocation:
            </p>
            <div
              className="grid gap-x-4 mt-2"
              style={{ grid: 'auto/auto auto 1fr ' }}
            >
              <h2>Now</h2>
              <h2 className="text-right">
                {formatPercentage(0)(stockAllocations.now)}
              </h2>
              <h2></h2>
              {!withdrawalsStarted && (
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

            <_ExpandableNote
              className="mt-4"
              title="Why the percentile for future years?"
            >
              <p className="p-base mb-4">
                Your stock allocation in the future may be a range and not a single
                number because the optimal allocation will depend on the
                relative sizes of competing spending goals and other resources
                like pensions. This depends on market performance and so we can get
                a range of possible allocations from the simulations.
              </p>
            </_ExpandableNote>
            {stockAllocations.atMaxAge > stockAllocations.atRetirement && (
              <_ExpandableNote
                className=""
                title="Why does the stock allocation increase between retirement and max age?"
              >
                <p className="p-base">
                  This happens when you have a legacy goal. As you get older,
                  more of your assets are going towards legacy and less towards
                  funding your remaining retirement years. Since you have a
                  higher risk tolerance for legacy, your portfolio becomes
                  correspondingly more aggressive. You can change your risk
                  tolerance for legacy in advanced settings.
                </p>
              </_ExpandableNote>
            )}
          </div>

          <button
            className="mt-6 underline disabled:lighten-2"
            onClick={() => handleChange(defaultRisk.riskTolerance.at20)}
            disabled={
              defaultRisk.riskTolerance.at20 ===
              params.risk.tpaw.riskTolerance.at20
            }
          >
            Reset to Default
          </button>
        </div>
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
      <div className={`${className}`}>
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
        {open && children}
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
    const { params, setParams, paramsExt } = useSimulation()
    const { longerLivedPerson } = paramsExt
    const defaultRisk = useMemo(() => getDefaultPlanParams().risk.tpaw, [])
    const isModified =
      defaultRisk.riskTolerance.deltaAtMaxAge !==
      params.risk.tpaw.riskTolerance.deltaAtMaxAge
    const handleChange = (value: number) =>
      setParams((params) => {
        const clone = _.cloneDeep(params)
        clone.risk.tpaw.riskTolerance.deltaAtMaxAge = value
        return clone
      })
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
          {params.people.withPartner &&
            (longerLivedPerson === 'person1'
              ? `This calculation will be based on your age since you have the longer remaining lifespan.`
              : `This calculation will be based on your partner's age since your partner has the longer remaining lifespan.`)}
        </p>
        <SliderInput
          className={`-mx-3 mt-2 `}
          height={60}
          maxOverflowHorz={props.sizing.cardPadding}
          data={RISK_TOLERANCE_VALUES.DATA.map((x) => -x)}
          value={params.risk.tpaw.riskTolerance.deltaAtMaxAge}
          onChange={handleChange}
          format={(x) => (-x).toFixed(0)}
          ticks={(value, i) => (i % 10 === 0 ? 'large' : 'small')}
        />
        <button
          className="mt-6 underline disabled:lighten-2"
          onClick={() => handleChange(defaultRisk.riskTolerance.deltaAtMaxAge)}
          disabled={!isModified}
        >
          Reset to Default
        </button>
      </div>
    )
  },
)

const _TPAWLegacyRiskToleranceDeltaCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { params, setParams } = useSimulation()
    const defaultRisk = useMemo(() => getDefaultPlanParams().risk.tpaw, [])
    const isModified =
      defaultRisk.riskTolerance.forLegacyAsDeltaFromAt20 !==
      params.risk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20

    const handleChange = (value: number) =>
      setParams((params) => {
        const clone = _.cloneDeep(params)
        clone.risk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20 = value
        return clone
      })
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
          data={RISK_TOLERANCE_VALUES.DATA}
          value={params.risk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20}
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

const _TPAWSpendingTiltCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { params, setParams } = useSimulation()
    const defaultRisk = useMemo(() => getDefaultPlanParams().risk.tpaw, [])
    const isModified =
      defaultRisk.timePreference !== params.risk.tpaw.timePreference
    const handleChange = (value: number) =>
      setParams((params) => {
        const clone = _.cloneDeep(params)
        clone.risk.tpaw.timePreference = value
        return clone
      })
    return (
      <div
        className={`${className} params-card relative`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
        <h2 className="font-bold text-lg mb-2">Spending Tilt</h2>
        <p className="p-base">
          This lets you shift your spending between early and late retirement.
          To spend more in early retirement and less in late retirement, move
          the slider to the left. To spend more in late retirement and less in
          early retirement, move the slider to the right.
        </p>
        <SliderInput
          className={`-mx-3 mt-2 `}
          height={60}
          maxOverflowHorz={props.sizing.cardPadding}
          data={[...TIME_PREFERENCE_VALUES].reverse()}
          value={params.risk.tpaw.timePreference}
          onChange={(x) => handleChange(x)}
          format={(x) => formatPercentage(1)(-x)}
          ticks={(value, i) => (i % 10 === 0 ? 'large' : 'small')}
        />
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
