import { faCircle as faCircleRegular } from '@fortawesome/pro-regular-svg-icons'
import { faCircle as faCircleSelected } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  PLAN_PARAMS_CONSTANTS,
  PickType,
  PlanParams,
  assert,
  block,
  letIn,
  noCase,
} from '@tpaw/common'
import _ from 'lodash'
import { DateTime } from 'luxon'
import React, { useState } from 'react'
import { SimulationResult } from '../../../../../UseSimulator/Simulator/Simulator'
import { formatCurrency } from '../../../../../Utils/FormatCurrency'
import { formatPercentage } from '../../../../../Utils/FormatPercentage'
import { paddingCSSStyle } from '../../../../../Utils/Geometry'
import { SliderInput } from '../../../../Common/Inputs/SliderInput/SliderInput'
import { CenteredModal } from '../../../../Common/Modal/CenteredModal'
import { SimpleModalListbox } from '../../../../Common/Modal/SimpleModalListbox'
import { useIANATimezoneName } from '../../../PlanRootHelpers/WithNonPlanParams'
import {
  useSimulation,
  useSimulationResult,
} from '../../../PlanRootHelpers/WithSimulation'
import { mainPlanColors } from '../../UsePlanColors'
import { PlanInputModifiedBadge } from '../Helpers/PlanInputModifiedBadge'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from '../PlanInputBody/PlanInputBody'
import {
  useIsPlanInputBondVolatilityCardModified,
  useIsPlanInputExpectedReturnsCardModified,
  useIsPlanInputStockVolatilityCardModified,
} from './UseIsPlanInputExpectedReturnsCardModified'
import {
  getExpectedReturnTypeLabelInfo,
  getExpectedReturnCustomStockBaseLabel,
  getExpectedReturnCustomBondBaseLabel,
} from './GetExpectedReturnLabelInto'
import { PlanParamsNormalized } from '../../../../../UseSimulator/NormalizePlanParams/NormalizePlanParams'

type ExpectedReturnsForPlanning =
  PlanParams['advanced']['returnsStatsForPlanning']['expectedValue']['empiricalAnnualNonLog']

export const PlanInputExpectedReturnsAndVolatility = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    return (
      <PlanInputBody {...props}>
        <>
          <_ExpectedReturnsCard className="" props={props} />
          <_StockVolatilityCard className="mt-10" props={props} />
          <_BondVolatilityCard className="mt-10" props={props} />
        </>
      </PlanInputBody>
    )
  },
)

type _DefaultData = {
  fixedEquityPremium: PickType<
    ExpectedReturnsForPlanning,
    'fixedEquityPremium'
  > | null
  custom: PickType<ExpectedReturnsForPlanning, 'custom'> | null
  fixed: PickType<ExpectedReturnsForPlanning, 'fixed'> | null
}
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
      defaultPlanParams,
      planParamsNorm,
      simulationResult,
    } = useSimulation()
    const { empiricalAnnualNonLog } =
      planParamsNorm.advanced.returnsStatsForPlanning.expectedValue

    const [defaultData, setDefaultData] = useState<_DefaultData>({
      fixedEquityPremium:
        empiricalAnnualNonLog.type === 'fixedEquityPremium'
          ? empiricalAnnualNonLog
          : null,
      custom:
        empiricalAnnualNonLog.type === 'custom' ? empiricalAnnualNonLog : null,
      fixed:
        empiricalAnnualNonLog.type === 'fixed' ? empiricalAnnualNonLog : null,
    })

    const handleChange = (value: ExpectedReturnsForPlanning) => {
      switch (value.type) {
        case 'regressionPrediction,20YearTIPSYield':
        case 'conservativeEstimate,20YearTIPSYield':
        case '1/CAPE,20YearTIPSYield':
        case 'historical':
          break
        case 'custom':
          setDefaultData((prev) => ({ ...prev, custom: value }))
          break
        case 'fixedEquityPremium':
          setDefaultData((prev) => ({ ...prev, fixedEquityPremium: value }))
          break
        case 'fixed':
          setDefaultData((prev) => ({ ...prev, fixed: value }))
          break
      }
      updatePlanParams('setExpectedReturnsForPlanning', value)
    }

    const isModified = useIsPlanInputExpectedReturnsCardModified()
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
            {`Pick the expected annual real returns for stocks and bonds. All the options other than "${letIn(
              getExpectedReturnTypeLabelInfo({ type: 'fixed' }),
              (labelInfo) => {
                assert(!labelInfo.isSplit)
                return labelInfo.stocksAndBonds.toLowerCase()
              },
            )}" are automatically updated periodically based on new data.`}
          </p>
          <div className="mt-6">
            <_Preset
              className="mt-4"
              type="regressionPrediction,20YearTIPSYield"
              onChange={handleChange}
            />
            <_Preset
              className="mt-4"
              type="conservativeEstimate,20YearTIPSYield"
              onChange={handleChange}
            />
            <_Preset
              className="mt-4"
              type="1/CAPE,20YearTIPSYield"
              onChange={handleChange}
            />
            <_FixedEquityPremium
              className="mt-4"
              onChange={handleChange}
              props={props}
              defaultData={
                defaultData.fixedEquityPremium ?? {
                  type: 'fixedEquityPremium',
                  equityPremium: 0.03,
                }
              }
            />
            <_Custom
              className="mt-4"
              onChange={handleChange}
              props={props}
              defaultData={
                defaultData.custom ?? {
                  type: 'custom',
                  stocks: { base: 'regressionPrediction', delta: 0 },
                  bonds: { base: '20YearTIPSYield', delta: 0 },
                }
              }
            />
            <_Preset
              className="mt-4"
              type="historical"
              onChange={handleChange}
            />
            <_Fixed
              className="mt-4"
              onChange={handleChange}
              props={props}
              defaultData={
                defaultData.fixed ?? {
                  type: 'fixed',
                  stocks: 0.05,
                  bonds: 0.02,
                }
              }
            />
          </div>

          <p className="mt-6 p-base">
            These presets were last updated on{' '}
            {getZonedTime(
              simulationResult.info.marketData.lastUpdatedAtTimestamp,
            ).toLocaleString(DateTime.DATE_MED)}
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
            handleChange(
              defaultPlanParams.advanced.returnsStatsForPlanning.expectedValue
                .empiricalAnnualNonLog,
            )
          }
          disabled={!isModified}
        >
          Reset to Default
        </button>
        <_ExpectedReturnsPresetPopup
          show={showCalculationPopup}
          onHide={() => setShowCalculationPopup(false)}
        />
      </div>
    )
  },
)

const _ExpectedReturnsPresetPopup = React.memo(
  ({ show, onHide }: { show: boolean; onHide: () => void }) => {
    const { simulationResult } = useSimulation()
    const marketData = simulationResult.info.marketData.expectedReturns
    const { getZonedTime } = useIANATimezoneName()

    const formatDate = (timestamp: number) =>
      getZonedTime(timestamp).toLocaleString(DateTime.DATE_MED)

    const getPresetLabelForStocks = (x: ExpectedReturnsForPlanning['type']) => {
      const labelInfo = getExpectedReturnTypeLabelInfo({ type: x })
      return labelInfo.isSplit ? labelInfo.stocks : labelInfo.stocksAndBonds
    }
    const getPresetLabelForBonds = (x: ExpectedReturnsForPlanning['type']) => {
      const labelInfo = getExpectedReturnTypeLabelInfo({ type: x })
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
            Data as of {formatDate(marketData.stocks.sp500.closingTime)}:
          </h2>
          <h2 className=" font-bold mt-4">CAPE Ratio</h2>
          <div className="p-base">
            {/* <p className="mt-3">CAPE calculation for the S&P 500 index:</p> */}
            <p className="mt-3 p-base">
              Price: The S&P 500 price as of{' '}
              {formatDate(marketData.stocks.sp500.closingTime)} was{' '}
              <span className="font-bold">
                {formatCurrency(marketData.stocks.sp500.value, 2)}
              </span>
              .
            </p>
            <p className="mt-3">
              Earnings: The ten year average real annual earnings of the S&P 500
              from{' '}
              {getZonedTime
                .fromObject({
                  year: marketData.stocks.averageRealEarningsForSP500For10Years
                    .tenYearDuration.start.year,
                  month:
                    marketData.stocks.averageRealEarningsForSP500For10Years
                      .tenYearDuration.start.month,
                })
                .toFormat('MMMM yyyy')}{' '}
              to{' '}
              {getZonedTime
                .fromObject({
                  year: marketData.stocks.averageRealEarningsForSP500For10Years
                    .tenYearDuration.end.year,
                  month:
                    marketData.stocks.averageRealEarningsForSP500For10Years
                      .tenYearDuration.end.month,
                })
                .toFormat('MMMM yyyy')}{' '}
              was{' '}
              <span className="font-bold">
                {formatCurrency(
                  marketData.stocks.averageRealEarningsForSP500For10Years.value,
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
                  {formatCurrency(marketData.stocks.sp500.value, 2)}
                </h2>
                <h2 className="text-center border-t border-gray-700">
                  {formatCurrency(
                    marketData.stocks.averageRealEarningsForSP500For10Years
                      .value,
                    2,
                  )}
                </h2>
              </div>
              <h2 className="">=</h2>
              <h2 className="text-center font-bold">
                {marketData.stocks.capeNotRounded.toFixed(2)}
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
                  {format(marketData.stocks.oneOverCAPERounded)}
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
                      {format(
                        marketData.stocks.empiricalAnnualNonLogRegressionsStocks
                          .full.fiveYear,
                      )}
                    </h2>{' '}
                    <h2 className="text-righ">10 year expected return: </h2>
                    <h2 className="font-bold">
                      {format(
                        marketData.stocks.empiricalAnnualNonLogRegressionsStocks
                          .full.tenYear,
                      )}
                    </h2>{' '}
                    <h2 className="text-righ">20 year expected return: </h2>
                    <h2 className="font-bold">
                      {format(
                        marketData.stocks.empiricalAnnualNonLogRegressionsStocks
                          .full.twentyYear,
                      )}
                    </h2>{' '}
                    <h2 className="text-righ">30 year expected return: </h2>
                    <h2 className="font-bold">
                      {format(
                        marketData.stocks.empiricalAnnualNonLogRegressionsStocks
                          .full.thirtyYear,
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
                      {format(
                        marketData.stocks.empiricalAnnualNonLogRegressionsStocks
                          .restricted.fiveYear,
                      )}
                    </h2>{' '}
                    <h2 className="text-righ">10 year expected return: </h2>
                    <h2 className="font-bold">
                      {format(
                        marketData.stocks.empiricalAnnualNonLogRegressionsStocks
                          .restricted.tenYear,
                      )}
                    </h2>{' '}
                    <h2 className="text-righ">20 year expected return: </h2>
                    <h2 className="font-bold">
                      {format(
                        marketData.stocks.empiricalAnnualNonLogRegressionsStocks
                          .restricted.twentyYear,
                      )}
                    </h2>{' '}
                    <h2 className="text-righ">30 year expected return: </h2>
                    <h2 className="font-bold">
                      {format(
                        marketData.stocks.empiricalAnnualNonLogRegressionsStocks
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
                  {format(marketData.stocks.regressionPrediction)}
                </span>{' '}
              </li>
              <li className="mt-1">
                {getPresetLabelForStocks(
                  'conservativeEstimate,20YearTIPSYield',
                )}{' '}
                — average of the four lowest of the nine core estimates above:{' '}
                <span className="font-bold">
                  {format(marketData.stocks.conservativeEstimate)}
                </span>{' '}
              </li>
              <li className="mt-1">
                {getPresetLabelForStocks('1/CAPE,20YearTIPSYield')}{' '}
                <span className="font-bold">
                  {format(marketData.stocks.oneOverCAPERounded)}
                </span>
              </li>
              <li className="mt-1">
                {getPresetLabelForStocks('historical')} — average of the
                historical stock returns:{' '}
                <span className="font-bold">
                  {format(marketData.stocks.historical)}
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
                Data as of {formatDate(marketData.bonds.bondRates.closingTime)}:
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
                  {format(marketData.bonds.bondRates.fiveYear)}
                </h2>
                <h2 className="">10 year TIPS yield: </h2>
                <h2 className="font-bold">
                  {format(marketData.bonds.bondRates.tenYear)}
                </h2>
                <h2 className="">20 year TIPS yield: </h2>
                <h2 className="font-bold">
                  {format(marketData.bonds.bondRates.twentyYear)}
                </h2>
                <h2 className="">30 year TIPS yield: </h2>
                <h2 className="font-bold">
                  {format(marketData.bonds.bondRates.thirtyYear)}
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
                    {format(marketData.bonds.bondRates.twentyYear)}
                  </span>{' '}
                </li>
                <li className="mt-1">
                  {getPresetLabelForBonds('historical')} — average of the
                  historical bond returns:{' '}
                  <span className="font-bold">
                    {format(marketData.bonds.historical)}
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

type _PresetType = Exclude<
  ExpectedReturnsForPlanning['type'],
  'fixed' | 'custom' | 'fixedEquityPremium'
>
type _CustomType = Extract<ExpectedReturnsForPlanning, { type: 'custom' }>

const _Preset = React.memo(
  ({
    className = '',
    type,
    onChange,
  }: {
    className?: string
    type: _PresetType
    onChange: (expected: ExpectedReturnsForPlanning) => void
  }) => {
    const { planParamsNorm, defaultPlanParams, simulationResult } =
      useSimulation()

    const marketData = simulationResult.info.marketData.expectedReturns
    const { stocks, bonds } = _resolveExpectedReturnPreset(type, marketData)
    const isDefault =
      defaultPlanParams.advanced.returnsStatsForPlanning.expectedValue
        .empiricalAnnualNonLog.type === type
    const labelInfo = getExpectedReturnTypeLabelInfo({ type })

    return (
      <button
        className={`${className} flex gap-x-2`}
        onClick={() => onChange({ type })}
      >
        <FontAwesomeIcon
          className="mt-1"
          icon={
            planParamsNorm.advanced.returnsStatsForPlanning.expectedValue
              .empiricalAnnualNonLog.type === type
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
              Stocks: {format(stocks)}, Bonds: {format(bonds)}
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
              Stocks: {format(stocks)}, Bonds: {format(bonds)}
            </h2>
          </div>
        )}
      </button>
    )
  },
)

export const _FixedEquityPremium = React.memo(
  ({
    className = '',
    onChange,
    props,
    defaultData,
  }: {
    className?: string
    onChange: (expected: ExpectedReturnsForPlanning) => void
    props: PlanInputBodyPassThruProps
    defaultData: PickType<ExpectedReturnsForPlanning, 'fixedEquityPremium'>
  }) => {
    const { planParamsNorm, simulationResult } = useSimulation()
    const marketData = simulationResult.info.marketData.expectedReturns
    const stocks =
      simulationResult.args.planParamsProcessed.returnsStatsForPlanning.stocks
        .empiricalAnnualNonLogExpectedReturnInfo.value
    const { empiricalAnnualNonLog } =
      planParamsNorm.advanced.returnsStatsForPlanning.expectedValue
    const labelInfo = getExpectedReturnTypeLabelInfo({
      type: 'fixedEquityPremium',
    })
    const bonds20YearTIPSYield = _resolveExpectedReturnBondsPreset(
      '20YearTIPSYield',
      marketData,
    )
    assert(!labelInfo.isSplit)

    return (
      <div className={`${className}`}>
        <button
          className={`${className} flex gap-x-2`}
          onClick={() => onChange(defaultData)}
        >
          <FontAwesomeIcon
            className="mt-1"
            icon={
              empiricalAnnualNonLog.type === 'fixedEquityPremium'
                ? faCircleSelected
                : faCircleRegular
            }
          />
          <div className="">
            <h2 className="text-start">{labelInfo.stocksAndBonds}</h2>
            <h2 className="text-start lighten-2 text-sm">
              Stocks: Bond Return + Fixed Equity Premium
            </h2>
            <h2 className="text-start lighten-2 text-sm">
              Bonds:{' '}
              {
                getExpectedReturnCustomBondBaseLabel('20YearTIPSYield')
                  .titleCase
              }
              : {format(bonds20YearTIPSYield)}
            </h2>
          </div>
        </button>
        {empiricalAnnualNonLog.type === 'fixedEquityPremium' && (
          <div className="mt-4 ml-6 bg-gray-100 rounded-lg px-4 py-4">
            <div className="col-span-2 flex justify-end">
              <h2 className="text-sm lighten-2 -mt-2 -mr-2 ">
                Stocks: {format(stocks)}, Bonds: {format(bonds20YearTIPSYield)}
              </h2>
            </div>
            <h2 className=" ">Fixed Equity Premium</h2>
            <SliderInput
              className="-mx-3"
              height={60}
              maxOverflowHorz={props.sizing.cardPadding}
              format={format}
              data={
                PLAN_PARAMS_CONSTANTS.advanced.returnsStatsForPlanning
                  .expectedValue.fixedEquityPremium.values
              }
              value={empiricalAnnualNonLog.equityPremium}
              onChange={(equityPremium) =>
                onChange({ type: 'fixedEquityPremium', equityPremium })
              }
              ticks={(__, i) =>
                i % 10 === 0
                  ? ('large' as const)
                  : i % 2 === 0
                    ? ('small' as const)
                    : ('none' as const)
              }
            />
            <p className="p-base mt-4">
              This corresponds to expected returns of:
            </p>
            <div
              className=" inline-grid gap-x-2  mt-2"
              style={{ grid: 'auto/auto auto' }}
            >
              <div
                className="inline-grid gap-x-2"
                style={{ grid: 'auto/auto auto auto' }}
              >
                <h2 className=""></h2>
                <h2 className="">Bonds</h2>
                <h2 className="">{format(bonds20YearTIPSYield)} </h2>
                <h2 className="">+</h2>
                <h2 className="mr-2">Fixed Equity Premium</h2>
                <h2 className="">
                  {format(empiricalAnnualNonLog.equityPremium)}{' '}
                </h2>
                <h2 className="">=</h2>
                <h2 className="">Stocks</h2>
                <h2 className="border-t border-gray-500">{format(stocks)}</h2>
              </div>
            </div>
          </div>
        )}
      </div>
    )
  },
)

export const _Custom = React.memo(
  ({
    className = '',
    onChange,
    props,
    defaultData,
  }: {
    className?: string
    onChange: (expected: ExpectedReturnsForPlanning) => void
    props: PlanInputBodyPassThruProps
    defaultData: PickType<ExpectedReturnsForPlanning, 'custom'>
  }) => {
    const { planParamsNorm, simulationResult } = useSimulation()
    const marketData = simulationResult.info.marketData.expectedReturns
    const { empiricalAnnualNonLog } =
      planParamsNorm.advanced.returnsStatsForPlanning.expectedValue

    const curr = {
      stocks:
        simulationResult.args.planParamsProcessed.returnsStatsForPlanning.stocks
          .empiricalAnnualNonLogExpectedReturnInfo.value,
      bonds:
        simulationResult.args.planParamsProcessed.returnsStatsForPlanning.bonds
          .empiricalAnnualNonLogExpectedReturnInfo.value,
    }
    return (
      <div className={`${className}`}>
        <button
          className={`${className} flex gap-x-2`}
          onClick={() => onChange(defaultData)}
        >
          <FontAwesomeIcon
            className="mt-1"
            icon={
              empiricalAnnualNonLog.type === 'custom'
                ? faCircleSelected
                : faCircleRegular
            }
          />
          <div className="">
            <h2 className="text-start">
              {letIn(
                getExpectedReturnTypeLabelInfo({ type: 'custom' }),
                (x) => {
                  assert(!x.isSplit)
                  return x.stocksAndBonds
                },
              )}
            </h2>
            <h2 className="text-start lighten-2 text-sm">
              Stocks: Preset + Fixed Delta
            </h2>
            <h2 className="text-start lighten-2 text-sm">
              Bonds: Preset + Fixed Delta
            </h2>
          </div>
        </button>
        {empiricalAnnualNonLog.type === 'custom' && (
          <div
            className="relative mt-4 ml-6 bg-gray-100 rounded-lg px-4 py-4 grid gap-x-4"
            style={{ grid: 'auto/auto 1fr' }}
          >
            <div className="col-span-2 flex justify-end">
              <h2 className="text-sm lighten-2 -mt-2 -mr-2 ">
                Stocks: {format(curr.stocks)}, Bonds: {format(curr.bonds)}
              </h2>
            </div>
            <h2 className=" col-span-2 font-bold  mb-2">Stocks</h2>
            <div className="col-span-2">
              <SimpleModalListbox
                className="py-1.5"
                value={empiricalAnnualNonLog.stocks.base}
                choices={
                  PLAN_PARAMS_CONSTANTS.advanced.returnsStatsForPlanning
                    .expectedValue.custom.stocks.base.values
                }
                onChange={(base) => {
                  onChange({
                    ...empiricalAnnualNonLog,
                    stocks: { ...empiricalAnnualNonLog.stocks, base },
                  })
                }}
                getLabel={(x) => {
                  const value = _resolveExpectedReturnStocksPreset(
                    x,
                    marketData,
                  )
                  const label =
                    getExpectedReturnCustomStockBaseLabel(x).titleCase

                  return `${label} (${format(value)})`
                }}
              />
            </div>
            <h2 className="py-3">Plus</h2>
            <SliderInput
              className="-mx-3"
              height={60}
              maxOverflowHorz={props.sizing.cardPadding}
              format={format}
              data={
                PLAN_PARAMS_CONSTANTS.advanced.returnsStatsForPlanning
                  .expectedValue.custom.deltaValues
              }
              value={empiricalAnnualNonLog.stocks.delta}
              onChange={(delta) =>
                onChange({
                  type: 'custom',
                  bonds: empiricalAnnualNonLog.bonds,
                  stocks: { ...empiricalAnnualNonLog.stocks, delta },
                })
              }
              ticks={(__, i) =>
                i % 10 === 0
                  ? ('large' as const)
                  : i % 2 === 0
                    ? ('small' as const)
                    : ('none' as const)
              }
            />
            <div className="col-span-2">
              <p className="p-base mt-2">
                This results in an expected return for stocks of{' '}
                {format(
                  _resolveExpectedReturnStocksPreset(
                    empiricalAnnualNonLog.stocks.base,
                    marketData,
                  ),
                )}{' '}
                + {format(empiricalAnnualNonLog.stocks.delta)} ={' '}
                {format(curr.stocks)}.
              </p>
            </div>
            <h2 className="col-span-2 mt-6 font-bold mb-2">Bonds</h2>
            <div className="col-span-2">
              <SimpleModalListbox
                className="py-1.5"
                value={empiricalAnnualNonLog.bonds.base}
                choices={
                  PLAN_PARAMS_CONSTANTS.advanced.returnsStatsForPlanning
                    .expectedValue.custom.bonds.base.values
                }
                onChange={(base) => {
                  onChange({
                    ...empiricalAnnualNonLog,
                    bonds: { ...empiricalAnnualNonLog.bonds, base },
                  })
                }}
                getLabel={(x) => {
                  const value = _resolveExpectedReturnBondsPreset(x, marketData)
                  const label =
                    getExpectedReturnCustomBondBaseLabel(x).titleCase
                  return `${label} (${format(value)})`
                }}
              />
            </div>
            <h2 className="py-3">Plus</h2>
            <SliderInput
              className="-mx-3"
              height={60}
              maxOverflowHorz={props.sizing.cardPadding}
              format={format}
              data={
                PLAN_PARAMS_CONSTANTS.advanced.returnsStatsForPlanning
                  .expectedValue.custom.deltaValues
              }
              value={empiricalAnnualNonLog.bonds.delta}
              onChange={(delta) =>
                onChange({
                  type: 'custom',
                  stocks: empiricalAnnualNonLog.stocks,
                  bonds: { ...empiricalAnnualNonLog.bonds, delta },
                })
              }
              ticks={(__, i) =>
                i % 10 === 0
                  ? ('large' as const)
                  : i % 2 === 0
                    ? ('small' as const)
                    : ('none' as const)
              }
            />
            <div className="col-span-2">
              <p className="p-base mt-2">
                This results in an expected return for bonds of{' '}
                {format(
                  _resolveExpectedReturnBondsPreset(
                    empiricalAnnualNonLog.bonds.base,
                    marketData,
                  ),
                )}{' '}
                + {format(empiricalAnnualNonLog.bonds.delta)} ={' '}
                {format(curr.bonds)}.
              </p>
            </div>
          </div>
        )}
      </div>
    )
  },
)

export const _Fixed = React.memo(
  ({
    className = '',
    onChange,
    props,
    defaultData,
  }: {
    className?: string
    onChange: (expected: ExpectedReturnsForPlanning) => void
    props: PlanInputBodyPassThruProps
    defaultData: PickType<ExpectedReturnsForPlanning, 'fixed'>
  }) => {
    const { planParamsNorm } = useSimulation()
    const { empiricalAnnualNonLog } =
      planParamsNorm.advanced.returnsStatsForPlanning.expectedValue

    return (
      <div className={`${className}`}>
        <button
          className={`${className} flex gap-x-2`}
          onClick={() => onChange(defaultData)}
        >
          <FontAwesomeIcon
            className="mt-1"
            icon={
              empiricalAnnualNonLog.type === 'fixed'
                ? faCircleSelected
                : faCircleRegular
            }
          />
          <div className="">
            <h2 className="text-start">
              {letIn(getExpectedReturnTypeLabelInfo({ type: 'fixed' }), (x) => {
                assert(!x.isSplit)
                return x.stocksAndBonds
              })}
            </h2>
          </div>
        </button>
        {empiricalAnnualNonLog.type === 'fixed' && (
          <div className="mt-4 ml-6 bg-gray-100 rounded-lg px-4 py-4">
            <div className="col-span-2 flex justify-end">
              <h2 className="text-sm lighten-2 -mt-2 -mr-2 ">
                Stocks: {format(empiricalAnnualNonLog.stocks)}, Bonds:{' '}
                {format(empiricalAnnualNonLog.bonds)}
              </h2>
            </div>
            <h2 className="mt-2 ">Stocks</h2>
            <SliderInput
              className="-mx-2"
              height={60}
              maxOverflowHorz={props.sizing.cardPadding}
              format={format}
              data={
                PLAN_PARAMS_CONSTANTS.advanced.returnsStatsForPlanning
                  .expectedValue.fixed.values
              }
              value={empiricalAnnualNonLog.stocks}
              onChange={(stocks) =>
                onChange({ ...empiricalAnnualNonLog, stocks })
              }
              ticks={(value, i) =>
                i % 10 === 0
                  ? ('large' as const)
                  : i % 2 === 0
                    ? ('small' as const)
                    : ('none' as const)
              }
            />

            <h2 className="">Bonds</h2>
            <SliderInput
              className="-mx-2"
              height={60}
              maxOverflowHorz={props.sizing.cardPadding}
              format={format}
              data={
                PLAN_PARAMS_CONSTANTS.advanced.returnsStatsForPlanning
                  .expectedValue.fixed.values
              }
              value={empiricalAnnualNonLog.bonds}
              onChange={(bonds) =>
                onChange({ ...empiricalAnnualNonLog, bonds })
              }
              ticks={(__, i) =>
                i % 10 === 0
                  ? ('large' as const)
                  : i % 2 === 0
                    ? ('small' as const)
                    : ('none' as const)
              }
            />
            <p className="p-base mt-4">
              Remember to use real and not nominal returns.
            </p>
          </div>
        )}
      </div>
    )
  },
)

const _resolveExpectedReturnPreset = (
  type: _PresetType,
  marketData: SimulationResult['info']['marketData']['expectedReturns'],
) => {
  const _resolve = (
    stocks: _CustomType['stocks']['base'],
    bonds: _CustomType['bonds']['base'],
  ) => ({
    stocks: _resolveExpectedReturnStocksPreset(stocks, marketData),
    bonds: _resolveExpectedReturnBondsPreset(bonds, marketData),
  })

  switch (type) {
    case 'regressionPrediction,20YearTIPSYield':
      return _resolve('regressionPrediction', '20YearTIPSYield')
    case 'conservativeEstimate,20YearTIPSYield':
      return _resolve('conservativeEstimate', '20YearTIPSYield')
    case '1/CAPE,20YearTIPSYield':
      return _resolve('1/CAPE', '20YearTIPSYield')
    case 'historical':
      return _resolve('historical', 'historical')
    default:
      noCase(type)
  }
}

const _resolveExpectedReturnStocksPreset = (
  type: _CustomType['stocks']['base'],
  marketData: SimulationResult['info']['marketData']['expectedReturns'],
) => {
  switch (type) {
    case 'regressionPrediction':
      return marketData.stocks.regressionPrediction
    case 'conservativeEstimate':
      return marketData.stocks.conservativeEstimate
    case '1/CAPE':
      return marketData.stocks.oneOverCAPERounded
    case 'historical':
      return marketData.stocks.historical
    default:
      noCase(type)
  }
}

const _resolveExpectedReturnBondsPreset = (
  type: _CustomType['bonds']['base'],
  marketData: SimulationResult['info']['marketData']['expectedReturns'],
) => {
  switch (type) {
    case '20YearTIPSYield':
      return marketData.bonds.tipsYield20Year
    case 'historical':
      return marketData.bonds.historical
    default:
      noCase(type)
  }
}

// Note that this is subtly different from the StockVolatilityCard. This does
// not control the volatility of "returnsStatsForPlanning" as is the case with
// the StockVolatilityCard. It controls the volatility of
// "historicalReturnsAdjusted". This is because bond volatility for
// "returnsStatsForPlanning" is always 0.
export const _BondVolatilityCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const {
      planParamsNorm,
      simulationResult,
      updatePlanParams,
      defaultPlanParams,
    } = useSimulation()
    const { planParamsProcessed } = simulationResult.args
    const isModified = useIsPlanInputBondVolatilityCardModified()
    const handleChange = (volatilityScale: number) => {
      updatePlanParams(
        'setHistoricalReturnsAdjustmentBondVolatilityScale',
        volatilityScale,
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
          The default simulation models bonds with volatility equal to the
          historical volatility. You can choose to model bonds with more or less
          volatility by choosing a scaling factor for the bond return
          distribution here. A scaling factor of{' '}
          <span className="italic">X</span> multiplies the deviations of
          historical log bond returns from its the mean by{' '}
          <span className="italic">X</span>. This will change the standard
          deviation to <span className="italic">X</span> times the historical
          standard deviation.
        </p>
        <SliderInput
          className={`-mx-3 mt-2 `}
          height={60}
          maxOverflowHorz={props.sizing.cardPadding}
          data={
            PLAN_PARAMS_CONSTANTS.advanced.historicalReturnsAdjustment
              .standardDeviation.bonds.scale.log.values
          }
          value={
            planParamsNorm.advanced.historicalReturnsAdjustment
              .standardDeviation.bonds.scale.log
          }
          onChange={(x) => handleChange(x)}
          format={(x) => x.toFixed(2)}
          ticks={(value, i) =>
            _.round(value, 1) === value ? 'large' : 'small'
          }
        />
        <p className="p-base mt-4">
          This corresponds to a standard deviation of log bond returns of{' '}
          {formatPercentage(2)(
            Math.sqrt(
              planParamsProcessed.historicalReturnsAdjusted.bonds.args
                .empiricalAnnualLogVariance,
            ),
          )}
          .
        </p>

        <button
          className="mt-6 underline disabled:lighten-2"
          onClick={() =>
            handleChange(
              defaultPlanParams.advanced.historicalReturnsAdjustment
                .standardDeviation.bonds.scale.log,
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
    const { planParamsNorm, updatePlanParams, defaultPlanParams } =
      useSimulation()
    const { planParamsProcessed } = useSimulationResult().args
    const isModified = useIsPlanInputStockVolatilityCardModified()
    const handleChange = (volatilityScale: number) => {
      updatePlanParams(
        'setReturnsStatsForPlanningStockVolatilityScale',
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
          The default simulation models stocks with volatility equal to the
          historical volatility. You can choose to model stocks with more or
          less volatility by choosing a scaling factor for the stock return
          distribution here. A scaling factor of{' '}
          <span className="italic">X</span> multiplies the deviations of
          historical log stock returns from its the mean by{' '}
          <span className="italic">X</span>. This will change the standard
          deviation to <span className="italic">X</span> times the historical
          standard deviation.
        </p>
        <SliderInput
          className={`-mx-3 mt-2 `}
          height={60}
          maxOverflowHorz={props.sizing.cardPadding}
          data={
            PLAN_PARAMS_CONSTANTS.advanced.returnsStatsForPlanning
              .standardDeviation.stocks.scale.log.values
          }
          value={
            planParamsNorm.advanced.returnsStatsForPlanning.standardDeviation
              .stocks.scale.log
          }
          onChange={(x) => handleChange(x)}
          format={(x) => x.toFixed(2)}
          ticks={(value, i) =>
            _.round(value, 1) === value ? 'large' : 'small'
          }
        />
        <p className="p-base mt-4">
          This corresponds to a standard deviation of log stock returns of{' '}
          {formatPercentage(2)(
            Math.sqrt(
              planParamsProcessed.returnsStatsForPlanning.stocks
                .empiricalAnnualLogVariance,
            ),
          )}
          .
        </p>

        <button
          className="mt-6 underline disabled:lighten-2"
          onClick={() =>
            handleChange(
              defaultPlanParams.advanced.returnsStatsForPlanning
                .standardDeviation.stocks.scale.log,
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

export const PlanInputExpectedReturnsAndVolatilitySummary = React.memo(
  ({ planParamsNorm }: { planParamsNorm: PlanParamsNormalized }) => {
    const { args, info } = useSimulationResult()
    const { empiricalAnnualNonLog } =
      planParamsNorm.advanced.returnsStatsForPlanning.expectedValue
    const marketData = info.marketData.expectedReturns

    const labelInfo = getExpectedReturnTypeLabelInfo(empiricalAnnualNonLog)
    const expectedReturns = {
      stocks:
        args.planParamsProcessed.returnsStatsForPlanning.stocks
          .empiricalAnnualNonLogExpectedReturnInfo.value,
      bonds:
        args.planParamsProcessed.returnsStatsForPlanning.bonds
          .empiricalAnnualNonLogExpectedReturnInfo.value,
    }
    const volatilityScale = {
      stocks:
        planParamsNorm.advanced.returnsStatsForPlanning.standardDeviation.stocks
          .scale.log,
      bonds:
        planParamsNorm.advanced.historicalReturnsAdjustment.standardDeviation
          .bonds.scale.log,
    }

    const volatilityAfterScaling = {
      stocks: Math.sqrt(
        args.planParamsProcessed.returnsStatsForPlanning.stocks
          .empiricalAnnualLogVariance,
      ),
      bonds: Math.sqrt(
        args.planParamsProcessed.historicalReturnsAdjusted.bonds.args
          .empiricalAnnualLogVariance,
      ),
    }

    return (
      <>
        <h2>
          Expected Returns
          {!labelInfo.isSplit ? `: ${labelInfo.stocksAndBonds}` : ''}
        </h2>
        {empiricalAnnualNonLog.type === 'fixedEquityPremium' ? (
          block(() => {
            assert(!labelInfo.isSplit)
            return (
              <div className="">
                <div
                  className="inline-grid gap-y-1 gap-x-2 ml-4"
                  style={{ grid: 'auto/auto auto 1fr' }}
                >
                  {/* <h2 className="ml-4">{labelInfo.stocksAndBonds}</h2> */}
                  <h2 className=""></h2>
                  <h2 className="mr-2">Bonds: 20 Year TIPS Yield</h2>
                  <h2 className="">{format(expectedReturns.bonds)}</h2>
                  <h2 className="">+</h2>
                  <h2 className="">Fixed Equity Premium:</h2>
                  <h2 className="">
                    {format(empiricalAnnualNonLog.equityPremium)}
                  </h2>
                  <h2 className="">=</h2>
                  <h2 className="">Stocks:</h2>
                  <h2 className="border-t border-gray-400 -mt-0.5 py-0.5">
                    {format(expectedReturns.stocks)}
                  </h2>
                </div>
              </div>
            )
          })
        ) : empiricalAnnualNonLog.type === 'custom' ? (
          block(() => {
            assert(!labelInfo.isSplit)
            return (
              <>
                {/* <h2 className="ml-4">{labelInfo.stocksAndBonds}</h2> */}

                <h2 className="ml-4">
                  Stocks:{' '}
                  {
                    getExpectedReturnCustomStockBaseLabel(
                      empiricalAnnualNonLog.stocks.base,
                    ).titleCase
                  }{' '}
                  (
                  {format(
                    _resolveExpectedReturnStocksPreset(
                      empiricalAnnualNonLog.stocks.base,
                      marketData,
                    ),
                  )}
                  ) + {format(empiricalAnnualNonLog.stocks.delta)} ={' '}
                  {format(expectedReturns.stocks)}
                </h2>
                <h2 className="ml-4">
                  Bonds:{' '}
                  {
                    getExpectedReturnCustomBondBaseLabel(
                      empiricalAnnualNonLog.bonds.base,
                    ).titleCase
                  }{' '}
                  (
                  {format(
                    _resolveExpectedReturnBondsPreset(
                      empiricalAnnualNonLog.bonds.base,
                      marketData,
                    ),
                  )}
                  ) + {format(empiricalAnnualNonLog.bonds.delta)} ={' '}
                  {format(expectedReturns.bonds)}
                </h2>
              </>
            )
          })
        ) : labelInfo.isSplit ? (
          <>
            <h2 className="ml-4">
              Stocks: {labelInfo.stocks}, {format(expectedReturns.stocks)}
            </h2>
            <h2 className="ml-4">
              Bonds: {labelInfo.bonds}, {format(expectedReturns.bonds)}
            </h2>
          </>
        ) : (
          <>
            {/* <h2 className="ml-4">{labelInfo.stocksAndBonds}</h2> */}
            <h2 className="ml-4">Stocks: {format(expectedReturns.stocks)}</h2>
            <h2 className="ml-4">Bonds: {format(expectedReturns.bonds)}</h2>
          </>
        )}
        <h2 className="">Stock Volatility</h2>
        <h2 className="ml-4">
          Scale by{' '}
          {`${volatilityScale.stocks.toFixed(2)}. Standard Deviation: ${formatPercentage(2)(volatilityAfterScaling.stocks)}`}
        </h2>
        <h2 className="">Bond Volatility</h2>
        <h2 className="ml-4">
          Scale by{' '}
          {`${volatilityScale.bonds.toFixed(2)}. Standard Deviation: ${formatPercentage(2)(volatilityAfterScaling.bonds)}`}
        </h2>
      </>
    )
  },
)

const format = formatPercentage(1)
