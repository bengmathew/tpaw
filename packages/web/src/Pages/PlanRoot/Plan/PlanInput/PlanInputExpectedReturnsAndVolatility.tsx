import { faCircle as faCircleRegular } from '@fortawesome/pro-regular-svg-icons'
import { faCircle as faCircleSelected } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  PlanParams,
  PLAN_PARAMS_CONSTANTS,
  assert,
  fGet,
  letIn,
  noCase,
} from '@tpaw/common'
import _ from 'lodash'
import { DateTime } from 'luxon'
import React, { useMemo, useState } from 'react'
import { CallRust } from '../../../../UseSimulator/PlanParamsProcessed/CallRust'
import { PlanParamsProcessed } from '../../../../UseSimulator/PlanParamsProcessed/PlanParamsProcessed'
import { formatCurrency } from '../../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../../Utils/FormatPercentage'
import { paddingCSSStyle } from '../../../../Utils/Geometry'
import { SliderInput } from '../../../Common/Inputs/SliderInput/SliderInput'
import { ToggleSwitch } from '../../../Common/Inputs/ToggleSwitch'
import { CenteredModal } from '../../../Common/Modal/CenteredModal'
import { useMarketData } from '../../PlanRootHelpers/WithMarketData'
import { useIANATimezoneName } from '../../PlanRootHelpers/WithNonPlanParams'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
import { mainPlanColors } from '../UsePlanColors'
import { PlanInputModifiedBadge } from './Helpers/PlanInputModifiedBadge'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from './PlanInputBody/PlanInputBody'
import { fWASM } from '../../../../UseSimulator/Simulator/GetWASM'
import * as Rust from '@tpaw/simulator'
import { PlanParamsNormalized } from '../../../../UseSimulator/NormalizePlanParams/NormalizePlanParams'

export const PlanInputExpectedReturnsAndVolatility = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    return (
      <PlanInputBody {...props}>
        <>
          <_ExpectedReturnsCard className="" props={props} />
          <_BondVolatilityCard className="mt-10" props={props} />
          <_StockVolatilityCard className="mt-10" props={props} />
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
    const {
      updatePlanParams,
      planParamsNorm,
      defaultPlanParams,
      currentMarketData,
    } = useSimulation()

    const data = useMemo(
      () =>
        fWASM().process_market_data_for_expected_returns_for_planning_presets(
          CallRust.getPlanParamsRust(planParamsNorm).advanced.sampling,
          planParamsNorm.advanced.historicalMonthlyLogReturnsAdjustment
            .standardDeviation,
          currentMarketData,
        ),
      [planParamsNorm, currentMarketData],
    )

    const handleChange = (
      value: PlanParams['advanced']['expectedReturnsForPlanning'],
    ) => updatePlanParams('setExpectedReturns2', value)

    const isModified = useIsExpectedReturnsCardModified()
    const [showCalculationPopup, setShowCalculationPopup] = useState(false)

    return (
      <div
        className={`${className} params-card relative`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />

        <h2 className="font-bold text-lg ">Expected Returns</h2>
        <div className="mt-2">
          <p className="p-base mb-2 mt-1">
            {`Pick the expected annual real returns for stocks and bonds. All the options other than "manual" are automatically updated periodically based on new data.`}
          </p>
          <div className="mt-6">
            <_Preset
              className="mt-4"
              type="regressionPrediction,20YearTIPSYield"
              onChange={handleChange}
              data={data}
            />
            <_Preset
              className="mt-4"
              type="conservativeEstimate,20YearTIPSYield"
              onChange={handleChange}
              data={data}
            />
            <_Preset
              className="mt-4"
              type="1/CAPE,20YearTIPSYield"
              onChange={handleChange}
              data={data}
            />
            <_Preset
              className="mt-4"
              type="historical"
              onChange={handleChange}
              data={data}
            />
          </div>
          <_Manual
            className="mt-4"
            onChange={handleChange}
            props={props}
            data={data}
          />
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
            handleChange(defaultPlanParams.advanced.expectedReturnsForPlanning)
          }
          disabled={!isModified}
        >
          Reset to Default
        </button>
        <_ExpectedReturnsPresetPopup
          show={showCalculationPopup}
          onHide={() => setShowCalculationPopup(false)}
          data={data}
        />
      </div>
    )
  },
)

const _ExpectedReturnsPresetPopup = React.memo(
  ({
    show,
    onHide,
    data,
  }: {
    show: boolean
    onHide: () => void
    data: Rust.DataForExpectedReturnsForPlanningPresets
  }) => {
    const { getZonedTime } = useIANATimezoneName()

    const formatDate = (timestamp: number) =>
      getZonedTime(timestamp).toLocaleString(DateTime.DATE_MED)

    const getPresetLabelForStocks = (
      x: PlanParams['advanced']['expectedReturnsForPlanning']['type'],
    ) => {
      const labelInfo = expectedReturnTypeLabelInfo({ type: x })
      return labelInfo.isSplit ? labelInfo.stocks : labelInfo.stocksAndBonds
    }
    const getPresetLabelForBonds = (
      x: PlanParams['advanced']['expectedReturnsForPlanning']['type'],
    ) => {
      const labelInfo = expectedReturnTypeLabelInfo({ type: x })
      return labelInfo.isSplit ? labelInfo.bonds : labelInfo.stocksAndBonds
    }

    return (
      <CenteredModal
        className=" dialog-outer-div"
        show={show}
        onOutsideClickOrEscape={onHide}
      >
        <div className="sm:p-4">
          <h2 className="text-2xl sm:text-3xl font-bold">Expected Returns</h2>
          <h2 className="text-xl sm:text-2xl font-bold mt-6">Stocks</h2>
          <p className="p-base mt-3">
            Earnings yield (E/P) measures such as 1/CAPE provide reasonable
            estimates of the expected real return of stocks. The estimates below
            are based on the CAPE ratio of the S&P 500 index, which is a broad
            measure of the US stock market.
          </p>
          <h2 className="p-base mt-3">
            Data as of {formatDate(data.stocks.sp500.closingTime)}:
          </h2>
          <h2 className=" font-bold mt-4">CAPE Ratio</h2>
          <div className="p-base">
            {/* <p className="mt-3">CAPE calculation for the S&P 500 index:</p> */}
            <p className="mt-3 p-base">
              Price: The S&P 500 price as of{' '}
              {formatDate(data.stocks.sp500.closingTime)} was{' '}
              <span className="font-bold">
                {formatCurrency(data.stocks.sp500.value, 2)}
              </span>
              .
            </p>
            <p className="mt-3">
              Earnings: The ten year average real annual earnings of the S&P 500
              from{' '}
              {getZonedTime
                .fromObject({
                  year: data.stocks.averageRealEarningsForSP500For10Years
                    .tenYearDuration.start.year,
                  month:
                    data.stocks.averageRealEarningsForSP500For10Years
                      .tenYearDuration.start.month,
                })
                .toFormat('MMMM yyyy')}{' '}
              to{' '}
              {getZonedTime
                .fromObject({
                  year: data.stocks.averageRealEarningsForSP500For10Years
                    .tenYearDuration.end.year,
                  month:
                    data.stocks.averageRealEarningsForSP500For10Years
                      .tenYearDuration.end.month,
                })
                .toFormat('MMMM yyyy')}{' '}
              was{' '}
              <span className="font-bold">
                {formatCurrency(
                  data.stocks.averageRealEarningsForSP500For10Years.value,
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
                  {formatCurrency(data.stocks.sp500.value, 2)}
                </h2>
                <h2 className="text-center border-t border-gray-700">
                  {formatCurrency(
                    data.stocks.averageRealEarningsForSP500For10Years.value,
                    2,
                  )}
                </h2>
              </div>
              <h2 className="">=</h2>
              <h2 className="text-center font-bold">
                {data.stocks.capeNotRounded.toFixed(2)}
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
                  {formatPercentage(1)(data.stocks.oneOverCAPERounded)}
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
                      {formatPercentage(1)(
                        data.stocks.empiricalAnnualNonLogRegressionsStocks.full
                          .fiveYear,
                      )}
                    </h2>{' '}
                    <h2 className="text-righ">10 year expected return: </h2>
                    <h2 className="font-bold">
                      {formatPercentage(1)(
                        data.stocks.empiricalAnnualNonLogRegressionsStocks.full
                          .tenYear,
                      )}
                    </h2>{' '}
                    <h2 className="text-righ">20 year expected return: </h2>
                    <h2 className="font-bold">
                      {formatPercentage(1)(
                        data.stocks.empiricalAnnualNonLogRegressionsStocks.full
                          .twentyYear,
                      )}
                    </h2>{' '}
                    <h2 className="text-righ">30 year expected return: </h2>
                    <h2 className="font-bold">
                      {formatPercentage(1)(
                        data.stocks.empiricalAnnualNonLogRegressionsStocks.full
                          .thirtyYear,
                      )}
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
                        data.stocks.empiricalAnnualNonLogRegressionsStocks
                          .restricted.fiveYear,
                      )}
                    </h2>{' '}
                    <h2 className="text-righ">10 year expected return: </h2>
                    <h2 className="font-bold">
                      {formatPercentage(1)(
                        data.stocks.empiricalAnnualNonLogRegressionsStocks
                          .restricted.tenYear,
                      )}
                    </h2>{' '}
                    <h2 className="text-righ">20 year expected return: </h2>
                    <h2 className="font-bold">
                      {formatPercentage(1)(
                        data.stocks.empiricalAnnualNonLogRegressionsStocks
                          .restricted.twentyYear,
                      )}
                    </h2>{' '}
                    <h2 className="text-righ">30 year expected return: </h2>
                    <h2 className="font-bold">
                      {formatPercentage(1)(
                        data.stocks.empiricalAnnualNonLogRegressionsStocks
                          .restricted.thirtyYear,
                      )}
                    </h2>{' '}
                  </div>
                </div>
              </div>
            </div>
            <h2 className=" font-bold mt-4">Presets</h2>
            <p className="p-base mt-3">The presets provided are as follows:</p>
            <ul className="list-disc ml-5 mt-2 p-base">
              <li className="mt-1">
                {getPresetLabelForStocks(
                  'regressionPrediction,20YearTIPSYield',
                )}{' '}
                — average of the eight regression based core estimates above:{' '}
                <span className="font-bold">
                  {formatPercentage(1)(data.stocks.regressionPrediction)}
                </span>{' '}
              </li>
              <li className="mt-1">
                {getPresetLabelForStocks(
                  'conservativeEstimate,20YearTIPSYield',
                )}{' '}
                — average of the four lowest of the nine core estimates above:{' '}
                <span className="font-bold">
                  {formatPercentage(1)(data.stocks.conservativeEstimate)}
                </span>{' '}
              </li>
              <li className="mt-1">
                {getPresetLabelForStocks('1/CAPE,20YearTIPSYield')}{' '}
                <span className="font-bold">
                  {formatPercentage(1)(data.stocks.oneOverCAPERounded)}
                </span>
              </li>
              <li className="mt-1">
                {getPresetLabelForStocks('historical')} — average of the
                historical stock returns:{' '}
                <span className="font-bold">
                  {formatPercentage(1)(data.stocks.historical)}
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
                Data as of {formatDate(data.bonds.bondRates.closingTime)}:
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
                  {formatPercentage(1)(data.bonds.bondRates.fiveYear)}
                </h2>
                <h2 className="">10 year TIPS yield: </h2>
                <h2 className="font-bold">
                  {formatPercentage(1)(data.bonds.bondRates.tenYear)}
                </h2>
                <h2 className="">20 year TIPS yield: </h2>
                <h2 className="font-bold">
                  {formatPercentage(1)(data.bonds.bondRates.twentyYear)}
                </h2>
                <h2 className="">30 year TIPS yield: </h2>
                <h2 className="font-bold">
                  {formatPercentage(1)(data.bonds.bondRates.thirtyYear)}
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
                  {getPresetLabelForBonds(
                    'regressionPrediction,20YearTIPSYield',
                  )}
                  :{' '}
                  <span className="font-bold">
                    {formatPercentage(1)(data.bonds.bondRates.twentyYear)}
                  </span>{' '}
                </li>
                <li className="mt-1">
                  {getPresetLabelForBonds('historical')} — average of the
                  historical bond returns:{' '}
                  <span className="font-bold">
                    {formatPercentage(1)(data.bonds.historical)}
                  </span>{' '}
                </li>
              </ul>
            </div>
          </div>
        </div>
      </CenteredModal>
    )
  },
)

const _Preset = React.memo(
  ({
    className = '',
    type,
    onChange,
    data,
  }: {
    className?: string
    type: Exclude<
      PlanParams['advanced']['expectedReturnsForPlanning']['type'],
      'manual'
    >
    onChange: (
      expected: PlanParams['advanced']['expectedReturnsForPlanning'],
    ) => void
    data: Rust.DataForExpectedReturnsForPlanningPresets
  }) => {
    const { planParamsNorm, defaultPlanParams } = useSimulation()
    const { stocks, bonds } = _resolveExpectedReturnPreset(type, data)
    const isDefault =
      defaultPlanParams.advanced.expectedReturnsForPlanning.type === type
    const labelInfo = expectedReturnTypeLabelInfo({ type })

    return (
      <button
        className={`${className} flex gap-x-2`}
        onClick={() => onChange({ type })}
      >
        <FontAwesomeIcon
          className="mt-1"
          icon={
            planParamsNorm.advanced.expectedReturnsForPlanning.type === type
              ? faCircleSelected
              : faCircleRegular
          }
        />
        {labelInfo.isSplit ? (
          <div className="">
            <h2 className="text-start">
              Stocks: {labelInfo.stocks}{' '}
              {isDefault && (
                <span
                  className="hidden sm:inline-block px-2 bg-gray-200 rounded-full text-sm ml-2"
                  style={{
                    backgroundColor: mainPlanColors.shades.light[4].hex,
                  }}
                >
                  default
                </span>
              )}
            </h2>
            <h2 className="text-start">Bonds: {labelInfo.bonds}</h2>
            <h2 className="text-start lighten-2 text-sm">
              Stocks: {formatPercentage(1)(stocks)}, Bonds:{' '}
              {formatPercentage(1)(bonds)}
            </h2>
            {isDefault && (
              <h2 className="sm:hidden text-start">
                <span className="inline-block px-2 bg-gray-200 rounded-full text-sm">
                  default
                </span>
              </h2>
            )}
          </div>
        ) : (
          <div className="">
            <h2 className="text-start">{labelInfo.stocksAndBonds}</h2>
            <h2 className="text-start lighten-2 text-sm">
              Stocks: {formatPercentage(1)(stocks)}, Bonds:{' '}
              {formatPercentage(1)(bonds)}
            </h2>
          </div>
        )}
      </button>
    )
  },
)

export const _Manual = React.memo(
  ({
    className = '',
    onChange,
    props,
    data,
  }: {
    className?: string
    onChange: (
      expected: PlanParams['advanced']['expectedReturnsForPlanning'],
    ) => void
    props: PlanInputBodyPassThruProps
    data: Rust.DataForExpectedReturnsForPlanningPresets
  }) => {
    const { planParamsNorm } = useSimulation()

    const curr =
      planParamsNorm.advanced.expectedReturnsForPlanning.type === 'manual'
        ? { ...planParamsNorm.advanced.expectedReturnsForPlanning }
        : _resolveExpectedReturnPreset(
            planParamsNorm.advanced.expectedReturnsForPlanning.type,
            data,
          )

    const findClosest = (curr: number) =>
      fGet(
        _.minBy(
          PLAN_PARAMS_CONSTANTS.manualStocksBondsNonLogAnnualReturnsValues,
          (x) => Math.abs(x - curr),
        ),
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
              planParamsNorm.advanced.expectedReturnsForPlanning.type ===
              'manual'
                ? faCircleSelected
                : faCircleRegular
            }
          />
          <div className="">
            <h2 className="text-start">
              {letIn(expectedReturnTypeLabelInfo({ type: 'manual' }), (x) => {
                assert(!x.isSplit)
                return x.stocksAndBonds
              })}
            </h2>
          </div>
        </button>
        {planParamsNorm.advanced.expectedReturnsForPlanning.type ===
          'manual' && (
          <div className="mt-4">
            <h2 className="ml-6 mt-2 ">Stocks</h2>
            <SliderInput
              className=""
              height={60}
              maxOverflowHorz={props.sizing.cardPadding}
              format={formatPercentage(1)}
              data={
                PLAN_PARAMS_CONSTANTS.manualStocksBondsNonLogAnnualReturnsValues
              }
              value={planParamsNorm.advanced.expectedReturnsForPlanning.stocks}
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
              data={
                PLAN_PARAMS_CONSTANTS.manualStocksBondsNonLogAnnualReturnsValues
              }
              value={planParamsNorm.advanced.expectedReturnsForPlanning.bonds}
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

export const expectedReturnTypeLabelInfo = ({
  type,
}: {
  type: PlanParams['advanced']['expectedReturnsForPlanning']['type']
}):
  | { isSplit: true; stocks: string; bonds: string; forUndoRedo: string }
  | { isSplit: false; stocksAndBonds: string } => {
  const bonds = '20 Year TIPS Yield'
  const bondsForUndoRedo = '20 year TIPS yield'
  const withTIPSBonds = (stocks: string, stocksForUndoRedo: string) => ({
    isSplit: true as const,
    stocks,
    bonds: bonds,
    forUndoRedo: `${stocksForUndoRedo} for stocks and ${bondsForUndoRedo} for bonds`,
  })
  switch (type) {
    case 'conservativeEstimate,20YearTIPSYield':
      return withTIPSBonds('Conservative Estimate', 'conservative estimate')
    case '1/CAPE,20YearTIPSYield':
      return withTIPSBonds('1/CAPE', '1/CAPE')
    case 'regressionPrediction,20YearTIPSYield':
      return withTIPSBonds('Regression Prediction', 'regression prediction')
    case 'historical':
      return { isSplit: false, stocksAndBonds: 'Historical' }
    case 'manual':
      return { isSplit: false, stocksAndBonds: 'Manual' }
  }
}

const _resolveExpectedReturnPreset = (
  type: Exclude<
    PlanParams['advanced']['expectedReturnsForPlanning']['type'],
    'manual'
  >,
  data: Rust.DataForExpectedReturnsForPlanningPresets,
) => {
  switch (type) {
    case 'regressionPrediction,20YearTIPSYield':
      return {
        stocks: data.stocks.regressionPrediction,
        bonds: data.bonds.tipsYield20Year,
      }
    case 'conservativeEstimate,20YearTIPSYield':
      return {
        stocks: data.stocks.conservativeEstimate,
        bonds: data.bonds.tipsYield20Year,
      }
    case '1/CAPE,20YearTIPSYield':
      return {
        stocks: data.stocks.oneOverCAPERounded,
        bonds: data.bonds.tipsYield20Year,
      }
    case 'historical':
      return {
        stocks: data.stocks.historical,
        bonds: data.bonds.historical,
      }
    default:
      noCase(type)
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
    const { planParamsNorm, updatePlanParams, defaultPlanParams } =
      useSimulation()
    const isModified = useIsBondVolatilityCardModified()
    const handleChange = (enableVolatility: boolean) => {
      updatePlanParams(
        'setHistoricalBondReturnsAdjustmentEnableVolatility',
        enableVolatility,
      )
    }
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
            checked={
              planParamsNorm.advanced.historicalMonthlyLogReturnsAdjustment
                .standardDeviation.bonds.enableVolatility
            }
            setChecked={(enabled) => {
              updatePlanParams(
                'setHistoricalBondReturnsAdjustmentEnableVolatility',
                enabled,
              )
            }}
          />
        </div>
        {/* <p className="">This sets the bond volatility to {}</p> */}

        <button
          className="mt-6 underline disabled:lighten-2"
          onClick={() =>
            handleChange(
              defaultPlanParams.advanced.historicalMonthlyLogReturnsAdjustment
                .standardDeviation.bonds.enableVolatility,
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

export const _StockVolatilityCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const {
      planParamsNorm,
      planParamsProcessed,
      updatePlanParams,
      defaultPlanParams,
    } = useSimulation()
    const isModified = useIsStockVolatilityCardModified()
    const handleChange = (volatilityScale: number) => {
      updatePlanParams(
        'setHistoricalStockReturnsAdjustmentVolatilityScale',
        volatilityScale,
      )
    }
    return (
      <div
        className={`${className} params-card relative`}
        style={{ ...paddingCSSStyle(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
        <h2 className="font-bold text-lg ">Stock Volatility</h2>
        <p className="p-base mt-2">
          The default simulation models stocks with volatility equalling the
          historical volatility. You can choose to model stocks with more or
          less volatility by choosing a scaling factor for the stock return
          distribution here. A scaling factor of{' '}
          <span className="italic">X</span> multiplies the deviations of
          historical log stock returns from its the mean by{' '}
          <span className="italic">X</span>. This will change the standard
          deviation to <span className="italic">X</span> times the historical
          standard deviation.
        </p>
        {/* <p className="p-base mt-2">
          So an input of 1 keeps the log stock return distribution the same as
          historical. An input less than 1 tightens the distribution and results
          in a lower standard deviation. An input greater than 1 widens the
          distribution and results in a higher standard deviation.
        </p> */}

        <SliderInput
          className={`-mx-3 mt-2 `}
          height={60}
          maxOverflowHorz={props.sizing.cardPadding}
          data={PLAN_PARAMS_CONSTANTS.stockVolatilityScaleValues}
          value={
            planParamsNorm.advanced.historicalMonthlyLogReturnsAdjustment
              .standardDeviation.stocks.scale
          }
          onChange={(x) => handleChange(x)}
          format={(x) => `${x}`}
          ticks={(value, i) =>
            _.round(value, 1) === value ? 'large' : 'small'
          }
        />
        <p className="p-base mt-4">
          This corresponds to a standard deviation of log stock returns of{' '}
          {formatPercentage(2)(
            Math.sqrt(
              planParamsProcessed.historicalMonthlyReturnsAdjusted.stocks.stats
                .empiricalAnnualLogVariance,
            ),
          )}
          .
        </p>

        <button
          className="mt-6 underline disabled:lighten-2"
          onClick={() =>
            handleChange(
              defaultPlanParams.advanced.historicalMonthlyLogReturnsAdjustment
                .standardDeviation.stocks.scale,
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

export const useIsPlanInputExpectedReturnsAndVolatilityModified = () => {
  const isExpectedCardModified = useIsExpectedReturnsCardModified()
  const isBondVolatilityModified = useIsBondVolatilityCardModified()
  const isStockVolatilityModified = useIsStockVolatilityCardModified()

  return (
    isExpectedCardModified ||
    isBondVolatilityModified ||
    isStockVolatilityModified
  )
}

const useIsExpectedReturnsCardModified = () => {
  const { planParamsNorm, defaultPlanParams } = useSimulation()
  return !_.isEqual(
    defaultPlanParams.advanced.expectedReturnsForPlanning,
    planParamsNorm.advanced.expectedReturnsForPlanning,
  )
}
const useIsStockVolatilityCardModified = () => {
  const { planParamsNorm, defaultPlanParams } = useSimulation()
  return !_.isEqual(
    defaultPlanParams.advanced.historicalMonthlyLogReturnsAdjustment
      .standardDeviation.stocks.scale,
    planParamsNorm.advanced.historicalMonthlyLogReturnsAdjustment
      .standardDeviation.stocks.scale,
  )
}

const useIsBondVolatilityCardModified = () => {
  const { planParamsNorm, defaultPlanParams } = useSimulation()
  return !_.isEqual(
    defaultPlanParams.advanced.historicalMonthlyLogReturnsAdjustment
      .standardDeviation.bonds.enableVolatility,
    planParamsNorm.advanced.historicalMonthlyLogReturnsAdjustment
      .standardDeviation.bonds.enableVolatility,
  )
}

export const PlanInputExpectedReturnsAndVolatilitySummary = React.memo(
  ({
    planParamsNorm,
    planParamsProcessed,
  }: {
    planParamsNorm: PlanParamsNormalized
    planParamsProcessed: PlanParamsProcessed
  }) => {
    const { expectedReturnsForPlanning } = planParamsProcessed
    const { historicalMonthlyLogReturnsAdjustment } = planParamsNorm.advanced
    const format = formatPercentage(1)
    const labelInfo = expectedReturnTypeLabelInfo(
      planParamsNorm.advanced.expectedReturnsForPlanning,
    )

    return (
      <>
        <h2>Expected Returns</h2>
        {labelInfo.isSplit ? (
          <>
            <h2 className="ml-4">
              Stocks: {labelInfo.stocks},{' '}
              {format(
                expectedReturnsForPlanning.empiricalAnnualNonLogReturnInfo
                  .stocks.value,
              )}
            </h2>
            <h2 className="ml-4">
              Bonds: {labelInfo.bonds},{' '}
              {format(
                expectedReturnsForPlanning.empiricalAnnualNonLogReturnInfo.bonds
                  .value,
              )}
            </h2>
          </>
        ) : (
          <>
            <h2 className="ml-4">{labelInfo.stocksAndBonds}</h2>
            <h2 className="ml-4">
              Stocks:{' '}
              {format(
                expectedReturnsForPlanning.empiricalAnnualNonLogReturnInfo
                  .stocks.value,
              )}
              , Bonds:{' '}
              {format(
                expectedReturnsForPlanning.empiricalAnnualNonLogReturnInfo.bonds
                  .value,
              )}
            </h2>
          </>
        )}
        <h2 className="">Volatility</h2>
        <h2 className="ml-4">
          Allow Bond Volatility:{' '}
          {`${
            historicalMonthlyLogReturnsAdjustment.standardDeviation.bonds
              .enableVolatility
              ? 'Yes'
              : 'No'
          }`}
        </h2>
        <h2 className="ml-4">
          Scale Stock Volatility by{' '}
          {`${historicalMonthlyLogReturnsAdjustment.standardDeviation.stocks.scale}`}
        </h2>
      </>
    )
  },
)
