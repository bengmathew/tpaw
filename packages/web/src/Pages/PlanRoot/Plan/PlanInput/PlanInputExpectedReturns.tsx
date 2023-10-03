import { faCircle as faCircleRegular } from '@fortawesome/pro-regular-svg-icons'
import {
  faCaretDown,
  faCaretRight,
  faCircle as faCircleSelected,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  EXPECTED_ANNUAL_RETURN_PRESETS,
  MANUAL_STOCKS_BONDS_RETURNS_VALUES,
  PlanParams,
  fGet,
  historicalReturns,
} from '@tpaw/common'
import _ from 'lodash'
import { DateTime } from 'luxon'
import React, { useState } from 'react'
import { formatCurrency } from '../../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../../Utils/FormatPercentage'
import { paddingCSSStyle } from '../../../../Utils/Geometry'
import { SliderInput } from '../../../Common/Inputs/SliderInput/SliderInput'
import { ToggleSwitch } from '../../../Common/Inputs/ToggleSwitch'
import { CenteredModal } from '../../../Common/Modal/CenteredModal'
import { useMarketData } from '../../PlanRootHelpers/WithMarketData'
import { useIANATimezoneName } from '../../PlanRootHelpers/WithNonPlanParams'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
import { PlanInputModifiedBadge } from './Helpers/PlanInputModifiedBadge'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'

export const PlanInputExpectedReturns = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    const advancedCount = _.filter([useIsFixedBondsCardModified()]).length

    const [showAdvanced, setShowAdvanced] = useState(false)

    return (
      <PlanInputBody {...props}>
        <>
          <_ExpectedReturnsCard className="" props={props} />
          <button
            className="mt-6 text-start"
            onClick={() => setShowAdvanced((x) => !x)}
            style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
          >
            <div className="text-[20px] sm:text-[24px] font-bold text-left">
              Advanced{' '}
              <FontAwesomeIcon
                icon={showAdvanced ? faCaretDown : faCaretRight}
              />
            </div>
            <h2 className="">
              {advancedCount === 0 ? 'None' : `${advancedCount} modified`}
            </h2>
          </button>
          {showAdvanced && (
            <>
              <_BondVolatilityCard className="" props={props} />
            </>
          )}
        </>
      </PlanInputBody>
    )
  },
)

export const _ExpectedReturnsCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { getZonedTime } = useIANATimezoneName()
    const { updatePlanParams, defaultPlanParams, currentMarketData } =
      useSimulation()
    const { CAPE, bondRates } = currentMarketData
    const formatDate = (timestamp: number) =>
      getZonedTime(timestamp).toLocaleString(DateTime.DATE_MED)

    const handleChange = (
      expected: PlanParams['advanced']['annualReturns']['expected'],
    ) => updatePlanParams('setExpectedReturns', expected)

    const isModified = useIsExpectedReturnsCardModified()
    const [showCalculationPopup, setShowCalculationPopup] = useState(false)

    return (
      <div
        className={`${className} params-card relative`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
        <div className="">
          <p className="p-base mb-2 mt-1">
            {`Pick the expected annual real returns for stocks and bonds. All the options other than "manual" are automatically updated periodically based on new data.`}
          </p>
          <div className="mt-6">
            <_Preset
              className="mt-4s"
              type="suggested"
              onChange={handleChange}
            />
            <_Preset
              className="mt-4"
              type="oneOverCAPE"
              onChange={handleChange}
            />
            <_Preset
              className="mt-4"
              type="regressionPrediction"
              onChange={handleChange}
            />
            <_Preset
              className="mt-4"
              type="historical"
              onChange={handleChange}
            />
          </div>
          <_Manual className="mt-4" onChange={handleChange} props={props} />
          <p className="mt-6 p-base">
            These presets were last updated on{' '}
            {getZonedTime(currentMarketData.closingTime).toLocaleString(
              DateTime.DATE_MED,
            )}
            . To see the calculations and data used for the presets as well as
            other options for expected returns,{` `}
            <button
              className="underline"
              onClick={() => setShowCalculationPopup(true)}
            >
              click here.
            </button>
          </p>
        </div>
        <button
          className="mt-3 underline disabled:lighten-2"
          onClick={() =>
            handleChange(defaultPlanParams.advanced.annualReturns.expected)
          }
          disabled={!isModified}
        >
          Reset to Default
        </button>
        <CenteredModal
          className=" dialog-outer-div"
          show={showCalculationPopup}
          onOutsideClickOrEscape={() => setShowCalculationPopup(false)}
        >
          <div className="sm:p-4">
            <h2 className="text-2xl sm:text-3xl font-bold">Expected Returns</h2>
            <h2 className="text-xl sm:text-2xl font-bold mt-6">Stocks</h2>
            <p className="p-base mt-3">
              Earnings yield (E/P) measures such as 1/CAPE provide reasonable
              estimates of the expected real return of stocks. The estimates
              below are based on the CAPE ratio of the S&P 500 index, which is a
              broad measure of the US stock market.
            </p>
            <h2 className="p-base mt-3">
              Data as of {formatDate(CAPE.closingTime)}:
            </h2>
            <h2 className=" font-bold mt-4">CAPE Ratio</h2>
            <div className="p-base">
              {/* <p className="mt-3">CAPE calculation for the S&P 500 index:</p> */}
              <p className="mt-3 p-base">
                Price: The S&P 500 price as of {formatDate(CAPE.closingTime)}{' '}
                was{' '}
                <span className="font-bold">
                  {formatCurrency(CAPE.sp500, 2)}
                </span>
                .
              </p>
              <p className="mt-3">
                Earnings: The ten year average real annual earnings of the S&P
                500 from{' '}
                {getZonedTime
                  .fromObject({
                    year: CAPE.averageAnnualRealEarningsForSP500For10Years
                      .tenYearDuration.start.year,
                    month:
                      CAPE.averageAnnualRealEarningsForSP500For10Years
                        .tenYearDuration.start.month,
                  })
                  .toFormat('MMMM yyyy')}{' '}
                to{' '}
                {getZonedTime
                  .fromObject({
                    year: CAPE.averageAnnualRealEarningsForSP500For10Years
                      .tenYearDuration.end.year,
                    month:
                      CAPE.averageAnnualRealEarningsForSP500For10Years
                        .tenYearDuration.end.month,
                  })
                  .toFormat('MMMM yyyy')}{' '}
                was{' '}
                <span className="font-bold">
                  {formatCurrency(
                    CAPE.averageAnnualRealEarningsForSP500For10Years.value,
                    2,
                  )}
                </span>
                {`. This data was obtained from the "US Stock Markets 1871 -
                Present and CAPE Ratio" spreadsheet available on Professor
                Robert Shiller's `}
                <a
                  href="http://www.econ.yale.edu/~shiller/data.htm"
                  target="_blank"
                  rel="noreferrer"
                  className="underline"
                >
                  website
                </a>
                .
              </p>
              <div
                className="inline-grid mt-3 items-center gap-x-1.5 sm:gap-x-3"
                style={{
                  grid: 'auto /auto auto auto auto auto auto auto',
                }}
              >
                <h2 className="">CAPE ratio</h2>
                <h2 className="">=</h2>
                <div className="">
                  <h2 className="text-center">price</h2>
                  <h2 className="text-center border-t border-gray-700">
                    earnings
                  </h2>
                </div>
                <h2 className="">=</h2>
                <div className="">
                  <h2 className="text-center">
                    {formatCurrency(CAPE.sp500, 2)}
                  </h2>
                  <h2 className="text-center border-t border-gray-700">
                    {formatCurrency(
                      CAPE.averageAnnualRealEarningsForSP500For10Years.value,
                      2,
                    )}
                  </h2>
                </div>
                <h2 className="">=</h2>
                <h2 className="text-center font-bold">
                  {CAPE.value.toFixed(2)}
                </h2>
              </div>
            </div>
            <h2 className=" font-bold mt-4">
              Core Estimates Based on CAPE Ratio
            </h2>
            <div className="">
              <p className="p-base mt-3">
                The CAPE ratio is used to arrive at the following nine core
                estimates of expected return:
              </p>
              <div className="p-base list-disc mt-3">
                <div className="mt-3">
                  1/CAPE:{' '}
                  <span className="font-bold">
                    {formatPercentage(1)(CAPE.oneOverCAPE)}
                  </span>
                </div>
                <div className="mt-3">
                  Regression of future log stock returns on log 1/CAPE yields:{' '}
                  <div className="">
                    <div
                      className="inline-grid mt-1 gap-x-2"
                      style={{ grid: ' auto/ auto auto' }}
                    >
                      <h2 className="text-righ">5 year expected return: </h2>
                      <h2 className="font-bold">
                        {formatPercentage(1)(CAPE.regression.full.fiveYear)}
                      </h2>{' '}
                      <h2 className="text-righ">10 year expected return: </h2>
                      <h2 className="font-bold">
                        {formatPercentage(1)(CAPE.regression.full.tenYear)}
                      </h2>{' '}
                      <h2 className="text-righ">20 year expected return: </h2>
                      <h2 className="font-bold">
                        {formatPercentage(1)(CAPE.regression.full.twentyYear)}
                      </h2>{' '}
                      <h2 className="text-righ">30 year expected return: </h2>
                      <h2 className="font-bold">
                        {formatPercentage(1)(CAPE.regression.full.thirtyYear)}
                      </h2>{' '}
                    </div>
                  </div>
                </div>
                <div className="mt-3">
                  Restricting the regression to data from 1950 onwards yields:{' '}
                  <div className="">
                    <div
                      className="inline-grid mt-1 gap-x-2"
                      style={{ grid: ' auto/ auto auto' }}
                    >
                      <h2 className="text-righ">5 year expected return: </h2>
                      <h2 className="font-bold">
                        {formatPercentage(1)(
                          CAPE.regression.restricted.fiveYear,
                        )}
                      </h2>{' '}
                      <h2 className="text-righ">10 year expected return: </h2>
                      <h2 className="font-bold">
                        {formatPercentage(1)(
                          CAPE.regression.restricted.tenYear,
                        )}
                      </h2>{' '}
                      <h2 className="text-righ">20 year expected return: </h2>
                      <h2 className="font-bold">
                        {formatPercentage(1)(
                          CAPE.regression.restricted.twentyYear,
                        )}
                      </h2>{' '}
                      <h2 className="text-righ">30 year expected return: </h2>
                      <h2 className="font-bold">
                        {formatPercentage(1)(
                          CAPE.regression.restricted.thirtyYear,
                        )}
                      </h2>{' '}
                    </div>
                  </div>
                </div>
              </div>
              <h2 className=" font-bold mt-4">Presets</h2>
              <p className="p-base mt-3">
                The presets provided are as follows:
              </p>
              <ul className="list-disc ml-5 mt-2 p-base">
                <li className="mt-1">
                  Suggested — average of the four lowest of the nine core
                  estimates above:{' '}
                  <span className="font-bold">
                    {formatPercentage(1)(CAPE.suggested)}
                  </span>{' '}
                </li>
                <li className="mt-1">
                  1/CAPE:{' '}
                  <span className="font-bold">
                    {formatPercentage(1)(CAPE.oneOverCAPE)}
                  </span>
                </li>
                <li className="mt-1">
                  Regression Prediction — average of the eight regression based
                  core estimates above:{' '}
                  <span className="font-bold">
                    {formatPercentage(1)(CAPE.regressionAverage)}
                  </span>{' '}
                </li>
                <li className="mt-1">
                  Historical — average of the historical stock returns:{' '}
                  <span className="font-bold">
                    {formatPercentage(1)(
                      historicalReturns.monthly.annualStats.stocks
                        .expectedValue,
                    )}
                  </span>{' '}
                </li>
              </ul>
            </div>
            <h2 className="text-xl sm:text-2xl font-bold mt-10">Bonds</h2>
            <div className="">
              <div className="p-base">
                <p className="mt-3">
                  TIPS yields provide reasonable estimates of the expected real
                  return of bonds. The data is available at{' '}
                  <a
                    href="https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics"
                    target="_blank"
                    rel="noreferrer"
                    className="underline"
                  >
                    treasury.gov
                  </a>
                  {` under "Daily Treasury Par Real Yield Curve Rates." Make sure you
                  click on the Real Yield Curve series, not the similarly named
                  Yield Curve series.`}
                </p>
                <p className="mt-3">
                  Data as of {formatDate(bondRates.closingTime)}:
                </p>
              </div>

              <h2 className=" font-bold mt-4">
                Core Estimates Based on TIPS Yields
              </h2>
              <div className="mt-3 p-base">
                <div
                  className="inline-grid gap-x-2"
                  style={{ grid: 'auto/auto auto' }}
                >
                  <h2 className="">5 year TIPS yield: </h2>
                  <h2 className="font-bold">
                    {formatPercentage(1)(bondRates.fiveYear)}
                  </h2>
                  <h2 className="">10 year TIPS yield: </h2>
                  <h2 className="font-bold">
                    {formatPercentage(1)(bondRates.tenYear)}
                  </h2>
                  <h2 className="">20 year TIPS yield: </h2>
                  <h2 className="font-bold">
                    {formatPercentage(1)(bondRates.twentyYear)}
                  </h2>
                  <h2 className="">30 year TIPS yield: </h2>
                  <h2 className="font-bold">
                    {formatPercentage(1)(bondRates.thirtyYear)}
                  </h2>
                </div>
              </div>
              <h2 className=" font-bold mt-4">Presets</h2>
              <div className="p-base">
                <p className="p-base mt-3">
                  The presets provided are as follows:
                </p>
                <ul className="list-disc ml-5 mt-2 p-base">
                  <li className="mt-1">
                    Suggested — the 20 year TIPS yield:{' '}
                    <span className="font-bold">
                      {formatPercentage(1)(bondRates.twentyYear)}
                    </span>{' '}
                  </li>
                  <li className="mt-1">
                    Historical — average of the historical bond returns:{' '}
                    <span className="font-bold">
                      {formatPercentage(1)(
                        historicalReturns.monthly.annualStats.bonds
                          .expectedValue,
                      )}
                    </span>{' '}
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </CenteredModal>
      </div>
    )
  },
)

export const _Preset = React.memo(
  ({
    className = '',
    type,
    onChange,
  }: {
    className?: string
    type: Parameters<typeof EXPECTED_ANNUAL_RETURN_PRESETS>[0]
    onChange: (
      expected: PlanParams['advanced']['annualReturns']['expected'],
    ) => void
  }) => {
    const { planParams, currentMarketData } = useSimulation()
    const { stocks, bonds } = EXPECTED_ANNUAL_RETURN_PRESETS(
      type,
      currentMarketData,
    )

    return (
      <button
        className={`${className} flex gap-x-2`}
        onClick={() => onChange({ type })}
      >
        <FontAwesomeIcon
          className="mt-1"
          icon={
            planParams.advanced.annualReturns.expected.type === type
              ? faCircleSelected
              : faCircleRegular
          }
        />
        <div className="">
          <h2 className="text-start">{expectedReturnTypeLabel({ type })}</h2>
          <h2 className="text-start lighten-2 text-sm">
            Stocks: {formatPercentage(1)(stocks)}, Bonds:{' '}
            {formatPercentage(1)(bonds)}
          </h2>
        </div>
      </button>
    )
  },
)

export const _Manual = React.memo(
  ({
    className = '',
    onChange,
    props,
  }: {
    className?: string
    onChange: (
      expected: PlanParams['advanced']['annualReturns']['expected'],
    ) => void
    props: PlanInputBodyPassThruProps
  }) => {
    const { marketData } = useMarketData()
    const { planParams, currentMarketData } = useSimulation()

    const curr =
      planParams.advanced.annualReturns.expected.type === 'manual'
        ? { ...planParams.advanced.annualReturns.expected }
        : EXPECTED_ANNUAL_RETURN_PRESETS(
            planParams.advanced.annualReturns.expected.type,
            currentMarketData,
          )

    const findClosest = (curr: number) =>
      fGet(
        _.minBy(MANUAL_STOCKS_BONDS_RETURNS_VALUES, (x) => Math.abs(x - curr)),
      )

    return (
      <div className={`${className}`}>
        <button
          className={`${className} flex gap-x-2`}
          onClick={() =>
            onChange({
              type: 'manual',
              stocks: findClosest(curr.stocks),
              bonds: findClosest(curr.bonds),
            })
          }
        >
          <FontAwesomeIcon
            className="mt-1"
            icon={
              planParams.advanced.annualReturns.expected.type === 'manual'
                ? faCircleSelected
                : faCircleRegular
            }
          />
          <div className="">
            <h2 className="text-start">
              {expectedReturnTypeLabel({ type: 'manual' })}
            </h2>
          </div>
        </button>
        {planParams.advanced.annualReturns.expected.type === 'manual' && (
          <div className="mt-4">
            <h2 className="ml-6 mt-2 ">Stocks</h2>
            <SliderInput
              className=""
              height={60}
              maxOverflowHorz={props.sizing.cardPadding}
              format={formatPercentage(1)}
              data={MANUAL_STOCKS_BONDS_RETURNS_VALUES}
              value={planParams.advanced.annualReturns.expected.stocks}
              onChange={(stocks) =>
                onChange({
                  type: 'manual',
                  stocks,
                  bonds: findClosest(curr.bonds),
                })
              }
              ticks={(value, i) =>
                i % 10 === 0
                  ? ('large' as const)
                  : i % 2 === 0
                  ? ('small' as const)
                  : ('none' as const)
              }
            />

            <h2 className="ml-6">Bonds</h2>
            <SliderInput
              className=""
              height={60}
              maxOverflowHorz={props.sizing.cardPadding}
              format={formatPercentage(1)}
              data={MANUAL_STOCKS_BONDS_RETURNS_VALUES}
              value={planParams.advanced.annualReturns.expected.bonds}
              onChange={(bonds) =>
                onChange({
                  type: 'manual',
                  stocks: findClosest(curr.stocks),
                  bonds,
                })
              }
              ticks={(value, i) =>
                i % 10 === 0
                  ? ('large' as const)
                  : i % 2 === 0
                  ? ('small' as const)
                  : ('none' as const)
              }
            />
            <p className="p-base ml-6">
              Remember to use real and not nominal returns.
            </p>
          </div>
        )}
      </div>
    )
  },
)
export const expectedReturnTypeLabel = ({
  type,
}: {
  type: PlanParams['advanced']['annualReturns']['expected']['type']
}) => {
  switch (type) {
    case 'suggested':
      return 'Suggested'
    case 'oneOverCAPE':
      return '1/CAPE for Stocks'
    case 'regressionPrediction':
      return 'Regression Prediction for Stocks'
    case 'historical':
      return 'Historical'
    case 'manual':
      return 'Manual'
  }
}

export const _BondVolatilityCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { planParams, updatePlanParams } = useSimulation()
    const isModified = useIsFixedBondsCardModified()
    const currHistorical = planParams.advanced.annualReturns.historical.bonds
    const currIsFixed =
      currHistorical.type === 'fixed' &&
      currHistorical.value.type === 'expectedUsedForPlanning'
    return (
      <div
        className={`${className} params-card relative`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
        <h2 className="font-bold text-lg ">Bond Volatility</h2>
        <p className="p-base mt-2">
          The default simulation models bonds with volatility equalling the
          historical volatility. The historical volatility comes from interest
          rate risk. You may instead want to model bonds with no volatility,
          making bond returns deterministic. Bonds returns will then equal the
          expected return every year. This might be a better model of bond
          returns when bonds are duration matched to reduce interest rate risk,
          especially when they form a large portion of the portfolio.
        </p>

        <div className="flex items-center gap-x-4 mt-4">
          <h2 className="font-medium">Allow Bond Volatility</h2>
          <ToggleSwitch
            className=""
            checked={!currIsFixed}
            setChecked={(enabled) => {
              updatePlanParams(
                'setHistoricalReturnsBonds',
                !enabled
                  ? 'fixedToExpectedUsedForPlanning'
                  : 'adjustExpectedToExpectedUsedForPlanning',
              )
            }}
          />
        </div>
        <button
          className="mt-6 underline disabled:lighten-2"
          onClick={() =>
            updatePlanParams(
              'setHistoricalReturnsBonds',
              'adjustExpectedToExpectedUsedForPlanning',
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

export const useIsPlanInputExpectedReturnsModified = () => {
  const isExpectedCardModified = useIsExpectedReturnsCardModified()
  const isFixedBondsCardModified = useIsFixedBondsCardModified()

  return isExpectedCardModified || isFixedBondsCardModified
}

const useIsExpectedReturnsCardModified = () => {
  const { planParams, defaultPlanParams } = useSimulation()
  return !_.isEqual(
    defaultPlanParams.advanced.annualReturns.expected,
    planParams.advanced.annualReturns.expected,
  )
}
const useIsFixedBondsCardModified = () => {
  const { planParams, defaultPlanParams } = useSimulation()
  return !_.isEqual(
    defaultPlanParams.advanced.annualReturns.historical.bonds,
    planParams.advanced.annualReturns.historical.bonds,
  )
}

export const PlanInputExpectedReturnsSummary = React.memo(() => {
  const { planParams, currentMarketData } = useSimulation()
  const format = formatPercentage(1)
  const { stocks, bonds } =
    planParams.advanced.annualReturns.expected.type === 'manual'
      ? planParams.advanced.annualReturns.expected
      : EXPECTED_ANNUAL_RETURN_PRESETS(
          planParams.advanced.annualReturns.expected.type,
          currentMarketData,
        )
  return (
    <>
      <h2>
        {expectedReturnTypeLabel(planParams.advanced.annualReturns.expected)}
      </h2>
      <h2 className="ml-4">Stocks: {format(stocks)}</h2>
      <h2 className="ml-4">Bonds: {format(bonds)}</h2>
      {planParams.advanced.annualReturns.historical.bonds.type === 'fixed' &&
        planParams.advanced.annualReturns.historical.bonds.value.type ===
          'expectedUsedForPlanning' && <h2>Bond Volatility Disabled </h2>}
    </>
  )
})
