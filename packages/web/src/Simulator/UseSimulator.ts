import {
  assert,
  assertFalse,
  block,
  fGet,
  MarketData,
  PlanParams,
} from '@tpaw/common'
import { useEffect, useMemo, useRef, useState } from 'react'

import {
  normalizePlanParams,
  PlanParamsNormalized,
} from './NormalizePlanParams/NormalizePlanParams'

import * as Rust from '@tpaw/simulator'
import _ from 'lodash'
import { useMarketData } from '../Pages/PlanRoot/PlanRootHelpers/WithMarketData'
import { CalendarDayFns } from '../Utils/CalendarDayFns'
import { CallRust } from './PlanParamsProcessed/CallRust'
import { processPlanParams } from './PlanParamsProcessed/PlanParamsProcessed'
import {
  DailyMarketSeriesSrc,
  simulateOnServer,
  SimulateOnServerResult,
} from './SimulateOnServer/SimulateOnServer'
import { FirstMonthSavingsPortfolioDetail } from './Simulator/GetFirstMonthSavingsPortfolioDetail'
import {
  SimulationArgs,
  SimulationResult,
  Simulator,
} from './Simulator/Simulator'
import {
  getPortfolioBalanceEstimationCacheHandlerForDatelessPlan,
  usePortfolioBalanceEstimationCache,
} from './UsePortfolioBalanceEstimationCache'

const DEBOUNCE_TIME_MS = 500

// Allow rounding of currency back.

// TODO: Testing. Remove @tpaw/simulator from package.json
// TODO: Testing. Remove wasm support
export type SimulationResult2 = SimulateOnServerResult & {
  planParamsNormOfResult: PlanParamsNormalized
  dailyMarketSeriesSrcOfResult: DailyMarketSeriesSrc
  ianaTimezoneNameIfDatedPlanOfResult: string | null
  percentilesOfResult: { low: number; mid: number; high: number }
  numOfSimulationForMonteCarloSamplingOfResult: number
  randomSeedOfResult: number
}

export function useSimulator(
  planId: string,
  dailyMarketSeriesSrc: DailyMarketSeriesSrc,
  planParamsHistoryUpToActualCurrentTimestamp: {
    id: string
    params: PlanParams
  }[],
  simulationTimestamp: number,
  ianaTimezoneNameIfDatedPlan: string | null,
  percentiles: { low: number; mid: number; high: number },
  numOfSimulationForMonteCarloSampling: number,
  randomSeed: number,
  runTests: boolean,
) {
  const { marketData } = useMarketData()
  const now = Date.now()
  const [state, setState] = useState<
    | {
        isRunning: true
        simulationStartTimestamp: {
          countingFromThisSimulation: number
          countingFromTheFirstDebouncedSimulation: number
        }
        prevResult: SimulationResult2 | null
      }
    | { isRunning: false; result: SimulationResult2 }
  >({
    isRunning: true,
    simulationStartTimestamp: {
      countingFromThisSimulation: now,
      countingFromTheFirstDebouncedSimulation: now,
    },
    prevResult: null,
  })

  const planParamsHistoryUpToSimulationTimestamp = useMemo(() => {
    return planParamsHistoryUpToActualCurrentTimestamp.slice(
      0,
      _.sortedLastIndexBy<{ params: { timestamp: number } }>(
        planParamsHistoryUpToActualCurrentTimestamp,
        { params: { timestamp: simulationTimestamp } },
        (x) => x.params.timestamp,
      ),
    )
  }, [planParamsHistoryUpToActualCurrentTimestamp, simulationTimestamp])

  const lastHistoryItem = fGet(_.last(planParamsHistoryUpToSimulationTimestamp))
  const planParamsNorm = useMemo(
    () =>
      normalizePlanParams(lastHistoryItem.params, {
        timestamp: simulationTimestamp,
        calendarDay: lastHistoryItem.params.datingInfo.isDated
          ? CalendarDayFns.fromTimestamp(
              simulationTimestamp,
              fGet(ianaTimezoneNameIfDatedPlan),
            )
          : null,
      }),
    [lastHistoryItem.params, simulationTimestamp, ianaTimezoneNameIfDatedPlan],
  )

  const { applyCache } = usePortfolioBalanceEstimationCache(planId, false)

  const lastRunTimestampRef = useRef<number>(0)

  useEffect(() => {
    const abortController = new AbortController()
    const now = Date.now()
    const timeSinceLastRun = now - lastRunTimestampRef.current
    lastRunTimestampRef.current = now
    setState((prev) => ({
      isRunning: true,
      simulationStartTimestamp: {
        countingFromThisSimulation: now,
        countingFromTheFirstDebouncedSimulation: prev.isRunning
          ? prev.simulationStartTimestamp
              .countingFromTheFirstDebouncedSimulation
          : now,
      },
      prevResult: prev.isRunning ? prev.prevResult : prev.result,
    }))
    const timeout = window.setTimeout(
      () => {
        block(async () => {
          abortController.signal.throwIfAborted()
          const simulateOnServerResult = await simulateOnServer(
            abortController.signal,
            dailyMarketSeriesSrc,
            applyCache(
              planParamsHistoryUpToSimulationTimestamp,
              simulationTimestamp,
              ianaTimezoneNameIfDatedPlan
                ? (timestamp) =>
                    CalendarDayFns.fromTimestamp(
                      timestamp,
                      ianaTimezoneNameIfDatedPlan,
                    )
                : () => assertFalse(),
            ),
            [percentiles.low, percentiles.mid, percentiles.high],
            planParamsNorm,
            numOfSimulationForMonteCarloSampling,
            randomSeed,
          )
          abortController.signal.throwIfAborted()

          const threshold = new Map([
            ['default', { value: 0.000001, percentage: 0.000001 }],
            [
              'allocation (savings portfolio)',
              { value: 0.009, percentage: 0.000001 },
            ],
            ['allocation (total portfolio)', { value: 100, percentage: 100 }],
            ['withdrawals rate', { value: 100, percentage: 100 }],
          ])
          // const localResult = fGet(
          //   await _getLocalResult(
          //     simulateOnServerResult.portfolioBalanceEstimationByDated
          //       .currentBalance,
          //     planParamsNorm,
          //     numOfSimulationForMonteCarloSampling,
          //     randomSeed,
          //     _getMarketDataForLocal(
          //       dailyMarketSeriesSrc,
          //       simulateOnServerResult,
          //       marketData,
          //       planParamsNorm,
          //     ),
          //   ),
          // )

          // const originalMaxDiffInfo = _checkAgainstLocal(
          //   simulateOnServerResult,
          //   localResult,
          //   'detail',
          // )
          // console.log(
          //   `%cSIMULATION DIFF`,
          //   'color:blue;font-weight:bold;border:1px solid blue;border-radius:5px;padding:5px;font-size:1.5em',
          // )
          // console.log(
          //   `          %cOriginal`,
          //   'color:blue;opacity:0.5',
          //   lastHistoryItem.params,
          // )
          // _logMaxDiffInfo(originalMaxDiffInfo, threshold, 'always', 'detail')

          // TODO: Testing. Remove.
          if (runTests) {
            const _runVariation = async (args: { fn: VariationFn }) => {
              const x = {
                planParams: _.cloneDeep(lastHistoryItem.params),
                moveForwardToEndMinusYears: undefined,
                portfolioBalance: 0,
              }
              args.fn(x)
              const planParamsNorm = block(() => {
                const getDating = (offsetMonths: number) => {
                  const effectiveTimestamp =
                    simulationTimestamp +
                    offsetMonths * 30 * 24 * 60 * 60 * 1000
                  return {
                    timestamp: effectiveTimestamp,
                    calendarDay: ianaTimezoneNameIfDatedPlan
                      ? CalendarDayFns.fromTimestamp(
                          effectiveTimestamp,
                          ianaTimezoneNameIfDatedPlan,
                        )
                      : null,
                  }
                }
                const noOffset = normalizePlanParams(x.planParams, getDating(0))
                return x.moveForwardToEndMinusYears === undefined
                  ? noOffset
                  : normalizePlanParams(
                      x.planParams,
                      getDating(
                        noOffset.ages.simulationMonths.lastMonthAsMFN -
                          x.moveForwardToEndMinusYears * 12,
                      ),
                    )
              })
              const [serverResult, localResult] = await Promise.all([
                simulateOnServer(
                  abortController.signal,
                  dailyMarketSeriesSrc,
                  getPortfolioBalanceEstimationCacheHandlerForDatelessPlan(
                    x.portfolioBalance,
                  ),
                  [percentiles.low, percentiles.mid, percentiles.high],
                  planParamsNorm,
                  11,
                  randomSeed,
                ),
                _getLocalResult(
                  x.portfolioBalance,
                  planParamsNorm,
                  11,
                  randomSeed,
                  _getMarketDataForLocal(
                    dailyMarketSeriesSrc,
                    simulateOnServerResult,
                    marketData,
                    planParamsNorm,
                  ),
                ),
              ])
              abortController.signal.throwIfAborted()
              const maxDiffInfo = _checkAgainstLocal(
                serverResult,
                fGet(localResult),
                'noDetail',
                // 'detail',
              )
              return { serverResult, maxDiffInfo, planParamsNorm, ...x }
            }

            const tests = _getTests(
              simulateOnServerResult,
              planParamsNorm.datingInfo,
            )
            const results = [] as Awaited<ReturnType<typeof _runVariation>>[]
            const start = performance.now()
            const chunkLength = 100
            for (const [chunkIndex, chunk] of _.chunk(
              tests,
              chunkLength,
            ).entries()) {
              abortController.signal.throwIfAborted()
              const currResults = await Promise.all(chunk.map(_runVariation))
              results.push(...currResults)
              for (const [i, result] of currResults.entries()) {
                const test = chunk[i]
                const index = i + chunkIndex * chunkLength
                console.log(
                  `%c${index.toString().padStart(4)}/${tests.length}:         ${test.label}`,
                  'color:blue',
                  // result.planParams.advanced.sampling
                  // result.planParams.advanced.returnsStatsForPlanning.standardDeviation.stocks.scale.log,
                  // result.planParams.advanced.historicalReturnsAdjustment.standardDeviation.bonds.scale.log
                  // result.planParams.risk.spawAndSWR,
                  // result.planParams.risk.swr,
                  // result.planParams.advanced.strategy,
                  // result.planParams.adjustmentsToSpending.tpawAndSPAW,
                  // result.planParamsNorm.datingInfo.nowAsCalendarDay
                  // result.serverResult.portfolioBalanceEstimationByDated
                  //   .currentBalance,
                )
              }
            }
            for (const [i, result] of results
              .entries()
              .filter(([_, x]) =>
                Array.from(x.maxDiffInfo.keys()).some(
                  (label) =>
                    _compareToThreshold(x.maxDiffInfo, threshold, label)
                      .isAbove,
                ),
              )) {
              console.log(
                `%c${i.toString().padStart(4)}/${results.length}: ${tests[i].label}`,
                'color:blue',
                result.planParams,
              )
              _logMaxDiffInfo(result.maxDiffInfo, threshold, 'always', 'detail')
            }
            const durationSeconds = (performance.now() - start) / 1000
            console.log(
              `SUMMARY:\n    numTests:  ${tests.length}\n    timing: ${durationSeconds.toFixed(0)}s\n    rate: ${(tests.length / durationSeconds).toFixed(2)} per second`,
            )
          }

          abortController.signal.throwIfAborted()
          const result: SimulationResult2 = {
            ...simulateOnServerResult,
            // This is needed because when dealing with the result, we need to run
            // calculations based on the exact args of the result (eg, numMonths in
            // planParamsNorm) and not the latest one that is available on
            // SimulationInfo for which the result might still be pending. We could
            // put this in and args object, but adding "OfResult" to the name forces
            // us to think about the choice explicitly.
            planParamsNormOfResult: planParamsNorm,
            dailyMarketSeriesSrcOfResult: dailyMarketSeriesSrc,
            ianaTimezoneNameIfDatedPlanOfResult: ianaTimezoneNameIfDatedPlan,
            percentilesOfResult: percentiles,
            numOfSimulationForMonteCarloSamplingOfResult:
              numOfSimulationForMonteCarloSampling,
            randomSeedOfResult: randomSeed,
          }
          setState({ isRunning: false, result })
        }).catch((e) => {
          if (abortController.signal.aborted) return
          throw e
        })
      },
      Math.max(0, DEBOUNCE_TIME_MS - timeSinceLastRun),
    )

    return () => {
      window.clearTimeout(timeout)
      abortController.abort()
    }
  }, [
    planParamsHistoryUpToSimulationTimestamp,
    simulationTimestamp,
    ianaTimezoneNameIfDatedPlan,
    numOfSimulationForMonteCarloSampling,
    randomSeed,
    dailyMarketSeriesSrc,
    applyCache,
    planParamsNorm,
    percentiles,
    marketData,
    runTests,
    // TODO: Testing. Remove.
    lastHistoryItem.params,
  ])
  const isRunningInfo = useMemo(
    () =>
      state.isRunning
        ? ({
            isRunning: true,
            simulationStartTimestamp: state.simulationStartTimestamp,
          } as const)
        : ({ isRunning: false } as const),
    [state],
  )
  return {
    isRunningInfo,
    simulationResult: state.isRunning ? state.prevResult : state.result,
    planParamsId: lastHistoryItem.id,
    planParamsNorm,
  }
}

//______________________________________________________________________________
//
// TODO: Testing. Delete
//______________________________________________________________________________

type MaxDiffInfo = Map<
  string,
  { value: number; percentage: number | 'localIsLessThanOne' }
>
type ThresholdInfo = Map<string, { value: number; percentage: number }>

const _compareToThreshold = (
  maxDiffInfo: MaxDiffInfo,
  thresholdInfo: ThresholdInfo,
  label: string,
) => {
  const labels = Array.from(thresholdInfo.keys())
  const thresholdLabel = thresholdInfo.has(label)
    ? label
    : (labels.find((x) => label.startsWith(x)) ?? 'default')
  // const thresholdLabel = thresholdInfo.has(label) ? label : 'default'

  const threshold = fGet(thresholdInfo.get(thresholdLabel))
  const { value, percentage } = fGet(maxDiffInfo.get(label))
  const aboveBy = {
    value: value - threshold.value,
    percentage:
      percentage === 'localIsLessThanOne'
        ? ('localIsLessThanOne' as const)
        : percentage - threshold.percentage,
  }

  const isAbove =
    aboveBy.value < 0
      ? false
      : aboveBy.percentage === 'localIsLessThanOne'
        ? true
        : aboveBy.percentage > 0

  return {
    threshold,
    label,
    value,
    aboveBy,
    isAbove,
    thresholdLabel,
    percentage,
  }
}

const _logMaxDiffInfo = (
  maxDiffInfo: MaxDiffInfo,
  threshold: ThresholdInfo,
  type: 'always' | 'aboveThreshold',
  detail: 'detail' | 'noDetail',
) => {
  const [failures, passes] = _.partition(
    Array.from(maxDiffInfo.keys()).map((label) =>
      _compareToThreshold(maxDiffInfo, threshold, label),
    ),
    (x) => x.isAbove,
  )

  if (type === 'aboveThreshold' && failures.length === 0) return

  const getString = (x: (typeof failures)[number]) => {
    const { value, label, threshold, percentage, thresholdLabel } = x
    return `%c${value.toString().padStart(25)} (diff), ${percentage.toString().padStart(25)} (diff as %) - ${label.padEnd(40)} using threshold (${thresholdLabel}): diff: ${threshold.value}, diff as %:${threshold.percentage}`
  }
  const getStyle = (x: (typeof failures)[number]) =>
    x.isAbove
      ? 'font-weight:bold;color:#e11d48'
      : 'font-weight:bold;color:#16a34a'

  console.log(
    failures.map(getString).join('\n'),
    ...failures.map(getStyle),
    detail === 'detail' ? { failures, passes } : undefined,
  )
}

// const _mergeMaxDiffInfo = (a: MaxDiffInfo, b: MaxDiffInfo) => {
//   const result = new Map(a)
//   for (const [label, { value, percentage }] of b.entries()) {
//     result.set(label, {
//       value: Math.max(result.get(label)?.value ?? 0, value),
//       percentage: Math.max(result.get(label)?.percentage ?? 0, percentage),
//     })
//   }
//   return result
// }

const _getHeading = (detail: boolean) => (x: string) => {
  if (detail) {
    console.log(`%c${' '.repeat(30)}${x}`, 'color:blue')
  }
}

const _getDiff = (local: number | null, remote: number | null) => {
  if (local === null || remote === null) {
    assert(local === null && remote === null)
    return { value: 0, percentage: 0 }
  }
  if (local === Infinity || remote === Infinity) {
    assert(local === Infinity && remote === Infinity)
    return { value: 0, percentage: 0 }
  }
  const value = Math.abs(local - remote)
  const percentage = local < 1 ? ('localIsLessThanOne' as const) : value / local
  return { value, percentage }
}
const _processDiff = <
  T extends {
    maxDiff: { value: number; percentage: number | 'localIsLessThanOne' }
  },
>(
  x: T,
  label: string,
  indent: number,
  maxDiffInfo: MaxDiffInfo,
  detail: boolean,
) => {
  if (detail) {
    const [integer, decimal] = x.maxDiff.value.toFixed(15).split('.')
    let decimalParts = _.chunk([...(decimal ?? '')], 5)
      .map((x) => x.join(''))
      .join(' ')
    const num = `${integer}.${decimalParts}`
    console.log(
      `%c${num.padStart(25).padEnd(30)}%c${' '.repeat(indent * 4)}${label}`,
      'color:gray',
      'color:blue',
      x,
    )
  }
  maxDiffInfo.set(label, x.maxDiff)
}

const _getProcessValue =
  (maxDiffInfo: MaxDiffInfo, detail: boolean) =>
  (label: string, local: number, remote: number, indent: number) => {
    _processDiff(
      { local, remote, maxDiff: _getDiff(local, remote) },
      label,
      indent,
      maxDiffInfo,
      detail,
    )
  }

const _getProcessArray =
  (maxDiffInfo: MaxDiffInfo, detail: boolean) =>
  (
    label: string,
    local: (number | null)[],
    remote: (number | null)[],
    indent: number,
    debug = false,
  ) => {
    assert(local.length === remote.length)
    const diffs = local.map((l, i) => _getDiff(l, remote[i]))

    const maxDiff = _.maxBy(diffs, (x) => x.value)
    _processDiff(
      { local, remote, diffs, maxDiff: maxDiff ?? { value: 0, percentage: 0 } },
      label,
      indent,
      maxDiffInfo,
      detail,
    )
  }

const _checkAgainstLocal = (
  remoteResult: SimulateOnServerResult,
  localResult: SimulationResult,
  detail: 'detail' | 'noDetail',
) => {
  const maxDiffInfo = new Map<string, { value: number; percentage: number }>()
  const _processValue = _getProcessValue(maxDiffInfo, detail === 'detail')
  const _processArray = _getProcessArray(maxDiffInfo, detail === 'detail')
  const _heading = _getHeading(detail === 'detail')
  const _get50th = (x: { percentile: number; data: number[] }[]) =>
    fGet(x.find((x) => x.percentile === 50))?.data

  const localProcessed = localResult.args.planParamsProcessed
  const remoteProcessed = remoteResult.planParamsProcessed
  {
    let local = localProcessed.adjustmentsToSpending
    let remote = remoteProcessed.adjustmentsToSpending
    _processArray(
      'planParamsProcessed::adjustmentsToSpending',
      [local.tpawAndSPAW.legacy.external, local.tpawAndSPAW.legacy.target],
      [remote.tpawAndSpaw.legacy.external, remote.tpawAndSpaw.legacy.target],
      0,
    )
  }
  {
    let local = localProcessed.returnsStatsForPlanning
    let remote = remoteProcessed.returnsStatsForPlanning
    _processArray(
      'planParamsProcessed::returnsStatsForPlanning',
      [
        local.stocks.empiricalAnnualLogVariance,
        local.stocks.empiricalAnnualNonLogExpectedReturnInfo.value,
        local.bonds.empiricalAnnualLogVariance,
        local.bonds.empiricalAnnualNonLogExpectedReturnInfo.value,
      ],
      [
        remote.stocks.empiricalAnnualLogVariance,
        remote.stocks.empiricalAnnualNonLogExpectedReturn,
        remote.bonds.empiricalAnnualLogVariance,
        remote.bonds.empiricalAnnualNonLogExpectedReturn,
      ],
      0,
    )
  }
  {
    const local = localProcessed.historicalReturnsAdjusted
    const remote = remoteProcessed.historicalReturns
    const statsArray = (x: typeof local.stocks.srcAnnualizedStats.log) => [
      x.mean,
      x.n,
      x.standardDeviation,
      x.variance,
    ]
    const localArray = (x: typeof local.stocks) => [
      x.args.empiricalAnnualLogVariance,
      x.args.empiricalAnnualNonLogExpectedReturnInfo.value,
      ...statsArray(x.srcAnnualizedStats.log),
      ...statsArray(x.srcAnnualizedStats.nonLog),
      ...statsArray(x.stats.annualized.log),
      ...statsArray(x.stats.annualized.nonLog),
      ...statsArray(x.stats.log),
    ]
    const remoteArray = (x: typeof remote.stocks) => [
      x.args.empiricalAnnualLogVariance,
      x.args.empiricalAnnualNonLogExpectedReturn,
      ...statsArray(x.srcAnnualizedStats.log),
      ...statsArray(x.srcAnnualizedStats.nonLog),
      ...statsArray(x.stats.annualized.log),
      ...statsArray(x.stats.annualized.nonLog),
      ...statsArray(x.stats.log),
    ]

    _processArray(
      'planParamsProcessed::historicalReturnsAdjusted',
      [...localArray(local.stocks), ...localArray(local.bonds)],
      [...remoteArray(remote.stocks), ...remoteArray(remote.bonds)],
      0,
    )
  }
  {
    _heading(`planParamsProcessed::risk`)
    const local = localProcessed.risk
    const remote = remoteProcessed.risk
    _processArray(
      'riskToleranceByMfn',
      local.tpaw.fullGlidePath.map((x) => x.unclamped.riskTolerance),
      remote.tpaw.riskToleranceByMfn,
      1,
    )
    _processArray(
      'rraUnclampedIncludingPosInfinityByMfn',
      local.tpaw.fullGlidePath.map((x) => x.unclamped.rra),
      remote.tpaw.rraUnclampedIncludingPosInfinityByMfn,
      1,
    )
  }
  {
    const local = localProcessed.marketDataProcessed
    const remote = remoteProcessed.marketDataForPresets
    const _help = (x: {
      fiveYear: number
      tenYear: number
      twentyYear: number
      thirtyYear: number
    }) => [x.fiveYear, x.tenYear, x.twentyYear, x.thirtyYear]
    _processArray(
      'planParamsProcessed::marketDataForPresets',
      [
        local.inflation.suggestedAnnual,
        local.expectedReturns.stocks.capeNotRounded,
        local.expectedReturns.stocks.conservativeEstimate,
        local.expectedReturns.stocks.historical,
        local.expectedReturns.stocks.oneOverCAPENotRounded,
        local.expectedReturns.stocks.oneOverCAPERounded,
        local.expectedReturns.stocks.regressionPrediction,
        ..._help(
          local.expectedReturns.stocks.empiricalAnnualNonLogRegressionsStocks
            .full,
        ),
        ..._help(
          local.expectedReturns.stocks.empiricalAnnualNonLogRegressionsStocks
            .restricted,
        ),
        local.expectedReturns.bonds.historical,
        local.expectedReturns.bonds.tipsYield20Year,
      ],
      [
        remote.inflation.suggestedAnnual,
        remote.expectedReturns.stocks.capeNotRounded,
        remote.expectedReturns.stocks.conservativeEstimate,
        remote.expectedReturns.stocks.historical,
        remote.expectedReturns.stocks.oneOverCapeNotRounded,
        remote.expectedReturns.stocks.oneOverCapeRounded,
        remote.expectedReturns.stocks.regressionPrediction,
        ..._help(
          remote.expectedReturns.stocks.empiricalAnnualNonLogRegressionsStocks
            .full,
        ),
        ..._help(
          remote.expectedReturns.stocks.empiricalAnnualNonLogRegressionsStocks
            .restricted,
        ),
        remote.expectedReturns.bonds.historical,
        remote.expectedReturns.bonds.tipsYield20Year,
      ],
      0,
    )
  }
  {
    _processValue(
      'planParamsProcessed::inflation',
      localProcessed.inflation.annual,
      remoteProcessed.annualInflation,
      0,
    )
  }
  {
    _heading(`planParamsProcessed::byMonth`)
    const _group = (
      local: Rust.ProcessedValueForMonthRanges,
      remote: SimulateOnServerResult['planParamsProcessed']['amountTimed']['wealth']['futureSavings'],
      label: string,
    ) => {
      const localHash = new Map(
        local.byId.map(({ id, values }) => [id, values]),
      )
      try {
        assert(localHash.size === remote.byId.length)
      } catch (e) {
        console.log('data', { local, remote })
        throw e
      }
      _processArray(`${label}::total`, remote.total, local.total, 1)
      remote.byId.forEach((remote) => {
        _processArray(
          `${label}::${remote.id}`,
          remote.values,
          fGet(localHash.get(remote.id)),
          1,
        )
      })
    }
    _group(
      localProcessed.byMonth.wealth.futureSavings,
      remoteProcessed.amountTimed.wealth.futureSavings,
      'futureSavings',
    )
    _group(
      localProcessed.byMonth.wealth.incomeDuringRetirement,
      remoteProcessed.amountTimed.wealth.incomeDuringRetirement,
      'incomeDuringRetirement',
    )
    _group(
      localProcessed.byMonth.adjustmentsToSpending.extraSpending.essential,
      remoteProcessed.amountTimed.adjustmentsToSpending.extraSpending.essential,
      'extraSpending::essential',
    )
    _group(
      localProcessed.byMonth.adjustmentsToSpending.extraSpending.discretionary,
      remoteProcessed.amountTimed.adjustmentsToSpending.extraSpending
        .discretionary,
      'extraSpending::discretionary',
    )
  }
  {
    const asArray = (x: FirstMonthSavingsPortfolioDetail) => [
      x.start.balance, // 0
      x.contributions.total, // 1
      x.contributions.toWithdrawal, // 2
      x.contributions.toSavingsPortfolio, // 3
      x.afterContributions.balance, // 4
      x.withdrawals.regular, // 5
      x.withdrawals.essential, // 6
      x.withdrawals.discretionary, // 7
      x.withdrawals.total, // 8
      x.withdrawals.fromSavingsPortfolio, // 9
      x.withdrawals.fromContributions, // 10
      x.afterWithdrawals.allocation.stocks, // 11
      x.afterWithdrawals.balance, // 12
      x.contributionToOrWithdrawalFromSavingsPortfolio.type === 'contribution'
        ? x.contributionToOrWithdrawalFromSavingsPortfolio.contribution
        : -x.contributionToOrWithdrawalFromSavingsPortfolio.withdrawal, // 13
    ]
    _processArray(
      'firstMonthOfSomeRun',
      asArray(localResult.firstMonthOfSomeRun),
      asArray(remoteResult.firstMonthOfSomeRun),
      0,
    )
  }

  {
    if (localResult.args.planParamsNorm.advanced.strategy === 'TPAW') {
      _heading(`tpawApproxNetPresentValueForBalanceSheet`)

      const localTPAW = localProcessed.netPresentValue.tpaw
      const remoteTPAW = fGet(
        remoteResult.tpawApproxNetPresentValueForBalanceSheet,
      )

      const _group = (
        local: (typeof localTPAW)['wealth']['futureSavings'],
        remote: (typeof remoteTPAW)['futureSavings'],
        label: string,
      ) => {
        const localHash = new Map(
          local.byId.map(({ id, values }) => [id, values]),
        )
        assert(localHash.size === remote.length)
        _processArray(
          label,
          remote.map((x) => fGet(localHash.get(x.id)).withCurrentMonth),
          remote.map((x) => x.value),
          1,
        )
      }
      _group(
        localTPAW.wealth.futureSavings,
        remoteTPAW.futureSavings,
        'balanceSheet::futureSavings',
      )
      _group(
        localTPAW.wealth.incomeDuringRetirement,
        remoteTPAW.incomeDuringRetirement,
        'balanceSheet::incomeDuringRetirement',
      )
      _group(
        localTPAW.adjustmentsToSpending.extraSpending.essential,
        remoteTPAW.essentialExpenses,
        'balanceSheet::extraSpending::essential',
      )
      _group(
        localTPAW.adjustmentsToSpending.extraSpending.discretionary,
        remoteTPAW.discretionaryExpenses,
        'balanceSheet::extraSpending::discretionary',
      )
      _processValue(
        'balanceSheet::legacy',
        localTPAW.adjustmentsToSpending.legacy,
        remoteTPAW.legacyTarget,
        1,
      )
    }
  }

  {
    _processArray(
      'endingBalanceOfSavingsPortfolioByPercentile',
      localResult.endingBalanceOfSavingsPortfolioByPercentile.map(
        (x) => x.data,
      ),
      remoteResult.endingBalanceOfSavingsPortfolioByPercentile.map(
        (x) => x.data,
      ),
      0,
    )
  }
  {
    _processValue(
      'num_simulationsActual',
      localResult.numSimulationsActual,
      remoteResult.numSimulationsActual,
      0,
    )
  }
  {
    _processValue(
      'numRunsWithInsufficientFunds',
      localResult.numRunsWithInsufficientFunds,
      remoteResult.numRunsWithInsufficientFunds,
      0,
    )
  }
  if (localResult.args.planParamsNorm.advanced.strategy === 'TPAW') {
    _processArray(
      'tpawSpendingTilt',
      localResult.args.planParamsProcessed.risk.tpawAndSPAW.monthlySpendingTilt,
      remoteResult.tpawSpendingTilt.total.byPercentileByMonthsFromNow[1].data,
      0,
    )
  }
  {
    _heading(`graphs`)
    const _group = (
      l: typeof localResult.savingsPortfolio.withdrawals.essential,
      r: typeof remoteResult.savingsPortfolio.withdrawals.essential,
      label: string,
    ) => {
      _processArray(
        `${label}::total`,
        _get50th(l.total.byPercentileByMonthsFromNow),
        _get50th(r.total.byPercentileByMonthsFromNow),
        1,
      )
      const localHash = new Map(
        l.byId.map((x) => [x.id, x.byPercentileByMonthsFromNow]),
      )
      assert(localHash.size === r.byId.length)
      r.byId.forEach((x) => {
        _processArray(
          `${label}::${x.id}`,
          _get50th(fGet(localHash.get(x.id))),
          _get50th(x.byPercentileByMonthsFromNow),
          1,
        )
      })
    }
    _processArray(
      'portfolio balance',
      _get50th(
        localResult.savingsPortfolio.start.balance.byPercentileByMonthsFromNow,
      ),
      _get50th(
        remoteResult.savingsPortfolio.start.balance.byPercentileByMonthsFromNow,
      ),
      1,
    )
    _group(
      localResult.savingsPortfolio.withdrawals.essential,
      remoteResult.savingsPortfolio.withdrawals.essential,
      'withdrawals.essential',
    )
    _group(
      localResult.savingsPortfolio.withdrawals.discretionary,
      remoteResult.savingsPortfolio.withdrawals.discretionary,
      'withdrawals.discretionary',
    )
    _processArray(
      'withdrawals.regular',
      _get50th(
        localResult.savingsPortfolio.withdrawals.regular
          .byPercentileByMonthsFromNow,
      ),
      _get50th(
        remoteResult.savingsPortfolio.withdrawals.regular
          .byPercentileByMonthsFromNow,
      ),
      1,
    )
    _processArray(
      'withdrawals.total',
      _get50th(
        localResult.savingsPortfolio.withdrawals.total
          .byPercentileByMonthsFromNow,
      ),
      _get50th(
        remoteResult.savingsPortfolio.withdrawals.total
          .byPercentileByMonthsFromNow,
      ),
      1,
    )
    try {
      _processArray(
        'withdrawals rate',
        _get50th(
          localResult.savingsPortfolio.withdrawals.fromSavingsPortfolioRate
            .byPercentileByMonthsFromNow,
        ),
        _get50th(
          remoteResult.savingsPortfolio.withdrawals.fromSavingsPortfolioRate
            .byPercentileByMonthsFromNow,
        ),
        1,
      )
    } catch (e) {
      console.log('withdrawals rate', {
        local: _get50th(
          localResult.savingsPortfolio.withdrawals.fromSavingsPortfolioRate
            .byPercentileByMonthsFromNow,
        ),
        remote: _get50th(
          remoteResult.savingsPortfolio.withdrawals.fromSavingsPortfolioRate
            .byPercentileByMonthsFromNow,
        ),
      })
      throw e
    }
    _processArray(
      'allocation (savings portfolio)',
      _get50th(
        localResult.savingsPortfolio.afterWithdrawals.allocation.stocks
          .byPercentileByMonthsFromNow,
      ),
      _get50th(
        remoteResult.savingsPortfolio.afterWithdrawals.allocation.stocks
          .byPercentileByMonthsFromNow,
      ),
      1,
    )
    _processArray(
      'allocation (total portfolio)',
      _get50th(
        localResult.totalPortfolio.afterWithdrawals.allocation.stocks
          .byPercentileByMonthsFromNow,
      ),
      _get50th(
        remoteResult.totalPortfolio.afterWithdrawals.allocationOrZeroIfNoWealth
          .stocks.byPercentileByMonthsFromNow,
      ),
      1,
    )
  }
  return maxDiffInfo
}

// Use local market data except if src is "live", in which case use remote.
const _getMarketDataForLocal = (
  dailyMarketSeriesSrc: DailyMarketSeriesSrc,
  remoteResult: SimulateOnServerResult | null,
  localMarketDataSrc: MarketData.Data,
  planParamsNorm: PlanParamsNormalized,
): Rust.DataForMarketBasedPlanParamValues => {
  const index =
    _.sortedIndexBy<{ closingTime: number }>(
      localMarketDataSrc,
      { closingTime: planParamsNorm.datingInfo.timestampForMarketData },
      (x) => x.closingTime,
    ) - 1
  const localMarketData: Rust.DataForMarketBasedPlanParamValues = {
    ...localMarketDataSrc[index],
    timestampForMarketData: planParamsNorm.datingInfo.timestampForMarketData,
  }

  if (dailyMarketSeriesSrc.type === 'live') {
    return localMarketData
  }
  assert(remoteResult)
  const renameClosingTime = <T extends { closingTimestamp: number }>(
    x: T,
  ): T & { closingTime: number } => ({
    ...x,
    closingTime: x.closingTimestamp,
  })

  return {
    closingTime:
      remoteResult.planParamsProcessed.marketDataForPresets.sourceRounded
        .dailyMarketData.closingTimestamp,
    inflation: renameClosingTime(
      remoteResult.planParamsProcessed.marketDataForPresets.sourceRounded
        .dailyMarketData.inflation,
    ),
    sp500: renameClosingTime(
      remoteResult.planParamsProcessed.marketDataForPresets.sourceRounded
        .dailyMarketData.sp500,
    ),
    bondRates: renameClosingTime(
      remoteResult.planParamsProcessed.marketDataForPresets.sourceRounded
        .dailyMarketData.bondRates,
    ),
    timestampForMarketData: planParamsNorm.datingInfo.timestampForMarketData,
  }
}

const _getLocalResult = async (
  portfolioBalance: number,
  planParamsNorm: PlanParamsNormalized,
  numOfSimulationsForMonteCarloSampling: number,
  randomSeed: number,
  marketData: Rust.DataForMarketBasedPlanParamValues,
) => {
  const localArgs: SimulationArgs = {
    currentPortfolioBalanceAmount: portfolioBalance,
    planParamsRust: CallRust.getPlanParamsRust(planParamsNorm),
    marketData,
    planParamsNorm,
    planParamsProcessed: processPlanParams(planParamsNorm, marketData),
    numOfSimulationForMonteCarloSampling: numOfSimulationsForMonteCarloSampling,
    randomSeed,
  }
  return await getSimulatorSingleton().runSimulations(
    { canceled: false },
    localArgs,
  )
}

// Singleton so this is created only once for speedup.
let _singleton: Simulator | null = null
export const getSimulatorSingleton = () => {
  if (!_singleton) _singleton = new Simulator()
  return _singleton
}

type VariationFn = (x: {
  planParams: PlanParams
  moveForwardToEndMinusYears?: number
  portfolioBalance: number
}) => void

// ------- Timing -------
const _getTimingVariations = () => {
  const result = [] as { label: string; fn: VariationFn }[]
  const add = (label: string, fn: VariationFn) => {
    result.push({ label, fn })
  }
  add('1/2-Last 15 Years', (x) => (x.moveForwardToEndMinusYears = 15))
  add('2/2-Now', (x) => (x.moveForwardToEndMinusYears = undefined))
  return result
}

// ------- Portfolio Balance -------
const _getPortfolioBalanceVariations = (portfolioBalance: number) => {
  const result = [] as { label: string; fn: VariationFn }[]
  const add = (label: string, fn: VariationFn) => {
    result.push({ label, fn })
  }
  add('1/3-Balance=0', (x) => (x.portfolioBalance = 0))
  add('2/3-Balance=Current', (x) => (x.portfolioBalance = portfolioBalance))
  add('3/3-Balance=100M', (x) => (x.portfolioBalance = 100_000_000))
  return result
}

// -------Spending Floor-------
const _getSpendingFloorVariations = (
  minWithdrawals: number,
  maxWithdrawals: number,
) => {
  const result = [] as {
    label: string
    fn: VariationFn
    value: number | null
    isHigh: boolean
  }[]
  const add = (label: string, value: number | null, isHigh: boolean) => {
    result.push({
      label,
      fn: (x) =>
        (x.planParams.adjustmentsToSpending.tpawAndSPAW.monthlySpendingFloor =
          value),
      value,
      isHigh,
    })
  }
  add('1/3-Floor(none)', null, false)
  add('2/3-Floor(mid)', (minWithdrawals + maxWithdrawals) / 2, false)
  add('3/3-Floor(high)', maxWithdrawals * 10, true)
  return result
}

// -------Spending Ceiling-------
const _getSpendingCeilingVariations = (
  maxWithdrawals: number,
  spendingFloor: number | null,
) => {
  const result = [] as {
    label: string
    fn: VariationFn
    value: number | null
    skipOnHighFloor: boolean
  }[]
  const add = (
    label: string,
    value: number | null,
    skipOnHighFloor: boolean,
  ) => {
    result.push({
      label,
      fn: (x) =>
        (x.planParams.adjustmentsToSpending.tpawAndSPAW.monthlySpendingCeiling =
          value),
      value,
      skipOnHighFloor,
    })
  }

  const effectiveFloor = spendingFloor ?? 0
  add('1/3-Ceiling(none)', null, false)
  // add('2/4-Ceiling(high)', Math.max(effectiveFloor, maxWithdrawals * 1.1), true)
  add(
    '2/3-Ceiling(mid)',
    Math.max(effectiveFloor, (maxWithdrawals + effectiveFloor) / 2),
    true,
  )
  add('2/3-Ceiling(floor)', effectiveFloor, false)
  return result
}

// ------ Legacy -------
const _getLegacyVariations = () => {
  const result = [] as {
    label: string
    fn: VariationFn
    value: PlanParams['adjustmentsToSpending']['tpawAndSPAW']['legacy']
  }[]
  const add = (
    label: string,
    value: PlanParams['adjustmentsToSpending']['tpawAndSPAW']['legacy'],
  ) => {
    result.push({
      label,
      fn: (x) =>
        (x.planParams.adjustmentsToSpending.tpawAndSPAW.legacy = value),
      value,
    })
  }
  add('1/3-Legacy=0', { total: 0, external: {} })
  add('2/3-Legacy=1M', {
    external: {
      abc: {
        label: 'Legacy',
        amount: 100,
        nominal: true,
        id: 'abc',
        sortIndex: 0,
        colorIndex: 0,
      },
      def: {
        label: 'Legacy',
        amount: 200,
        nominal: false,
        id: 'def',
        sortIndex: 1,
        colorIndex: 1,
      },
    },
    total: 1000000,
  })
  add('3/3-Legacy=100M', {
    external: {},
    total: 100_000_000,
  })
  return result
}

// -------Market Performance-------
const _getMarketPerformanceVariations = () => {
  const result = [] as { label: string; fn: VariationFn }[]
  const add = (
    label: string,
    value: PlanParams['advanced']['historicalReturnsAdjustment']['overrideToFixedForTesting'],
  ) => {
    result.push({
      label,
      fn: (x) =>
        (x.planParams.advanced.historicalReturnsAdjustment.overrideToFixedForTesting =
          value),
    })
  }
  add('1/3-Market(high)', { type: 'manual', stocks: 0.2, bonds: 0.1 })
  add('2/3-Market(expected)', { type: 'useExpectedReturnsForPlanning' })
  add('3/3-Market(low)', { type: 'manual', stocks: -0.2, bonds: -0.1 })
  return result
}

// -------Strategy-------
const _getStrategyVariations = () => {
  const result = [] as { label: string; fn: VariationFn }[]
  const add = (label: string, fn: VariationFn) => {
    result.push({ label, fn })
  }

  {
    // TPAW
    for (const risk of ['zero', 'mid', 'max'] as const) {
      for (const addTilt of ['min', 'zero', 'max'] as const) {
        add(`TPAW (${risk} risk, ${addTilt} tilt)`, (x) => {
          x.planParams.advanced.strategy = 'TPAW'
          x.planParams.risk.tpaw.riskTolerance.at20 =
            risk === 'zero' ? 0 : risk === 'mid' ? 12 : 24
          x.planParams.risk.tpaw.additionalAnnualSpendingTilt =
            addTilt === 'min' ? -0.05 : addTilt === 'zero' ? 0 : 0.05
        })
      }
    }
  }

  {
    // SPAW
    for (const tilt of ['min', 'zero', 'max'] as const) {
      add(`SPAW (${tilt} tilt)`, (x) => {
        x.planParams.advanced.strategy = 'SPAW'
        x.planParams.risk.spaw.annualSpendingTilt =
          tilt === 'min' ? -0.03 : tilt === 'zero' ? 0 : 0.03
      })
    }
  }

  {
    // SWR
    for (const percentPerYear of [0, 0.04, 0.3]) {
      add(`SWR (${(percentPerYear * 100).toFixed(0)}%)`, (x) => {
        x.planParams.advanced.strategy = 'SWR'
        x.planParams.risk.swr.withdrawal = {
          type: 'asPercentPerYear',
          percentPerYear,
        }
      })
    }
    for (const amountPerMonth of [0, 10000]) {
      add(`SWR ($${amountPerMonth})`, (x) => {
        x.planParams.advanced.strategy = 'SWR'
        x.planParams.risk.swr.withdrawal = {
          type: 'asAmountPerMonth',
          amountPerMonth,
        }
      })
    }
  }
  return result
}

const _getNonCombinatorialTests = (
  datingInfo: PlanParamsNormalized['datingInfo'],
) => {
  const result = [] as { label: string; fn: VariationFn }[]
  const add = (label: string, fn: VariationFn) => {
    result.push({ label, fn })
  }
  {
    // TPAW
    for (const risk of ['min', 'mid', 'max'] as const) {
      for (const addTilt of ['min', 'zero', 'max'] as const) {
        for (const deltaMaxAge of ['min', 'default', 'max'] as const) {
          for (const deltaLegacy of ['min', 'default', 'max'] as const) {
            for (const prefFuture of ['min', 'default', 'max'] as const) {
              add(
                `TPAW ${risk} risk, ${addTilt} addTilt, ${deltaMaxAge} deltaMaxAge, ${deltaLegacy} deltaLegacy, ${prefFuture} prefFuture`,
                (x) => {
                  x.planParams.advanced.strategy = 'TPAW'
                  x.planParams.risk.tpaw.riskTolerance.at20 =
                    risk === 'min' ? 0 : risk === 'mid' ? 12 : 24
                  x.planParams.risk.tpaw.additionalAnnualSpendingTilt =
                    addTilt === 'min' ? -0.05 : addTilt === 'zero' ? 0 : 0.05
                  x.planParams.risk.tpaw.riskTolerance.deltaAtMaxAge =
                    deltaMaxAge === 'min'
                      ? 0
                      : deltaMaxAge === 'default'
                        ? -2
                        : -24
                  x.planParams.risk.tpaw.riskTolerance.forLegacyAsDeltaFromAt20 =
                    deltaLegacy === 'min'
                      ? 0
                      : deltaLegacy === 'default'
                        ? 2
                        : 24
                  x.planParams.risk.tpaw.timePreference =
                    prefFuture === 'min'
                      ? -0.05
                      : prefFuture === 'default'
                        ? 0
                        : 0.05
                },
              )
            }
          }
        }
      }
    }
  }

  const addAllGlidePaths = (label: string, fn: VariationFn) => {
    const edges = [0, 0.25, 0.5, 0.75, 1]
    for (const start of edges) {
      for (const end of edges) {
        add(`${label} glidepath=${start}-${end}`, (x) => {
          fn(x)
          x.planParams.risk.spawAndSWR.allocation = {
            start: {
              stocks: start,
              month: {
                type: 'now',
                monthOfEntry: datingInfo.isDated
                  ? {
                      isDatedPlan: true,
                      calendarMonth: {
                        year: datingInfo.nowAsCalendarDay.year,
                        month: datingInfo.nowAsCalendarDay.month,
                      },
                    }
                  : { isDatedPlan: false },
              },
            },
            intermediate: {},
            end: { stocks: end },
          }
        })
      }
    }
  }

  {
    // SPAW
    for (const tilt of ['min', 'zero', 'max'] as const) {
      addAllGlidePaths(`SPAW ${tilt} tilt`, (x) => {
        x.planParams.advanced.strategy = 'SPAW'
        x.planParams.risk.spaw.annualSpendingTilt =
          tilt === 'min' ? -0.03 : tilt === 'zero' ? 0 : 0.03
      })
    }
  }

  {
    // SWR
    for (const percentPerYear of [0, 0.04, 0.3]) {
      addAllGlidePaths(`SWR ${(percentPerYear * 100).toFixed(0)}%`, (x) => {
        x.planParams.advanced.strategy = 'SWR'
        x.planParams.risk.swr.withdrawal = {
          type: 'asPercentPerYear',
          percentPerYear,
        }
      })
    }
    for (const amountPerMonth of [0, 10000]) {
      addAllGlidePaths(`SWR $${amountPerMonth}`, (x) => {
        x.planParams.advanced.strategy = 'SWR'
        x.planParams.risk.swr.withdrawal = {
          type: 'asAmountPerMonth',
          amountPerMonth,
        }
      })
    }
  }

  {
    // Expected returns
    const values = [-0.01, 0.02, 0.5, 0.8, 0.1]
    for (const stocks of values) {
      for (const bonds of values) {
        add(
          `Expected returns (${stocks.toFixed(2)} stocks, ${bonds.toFixed(2)} bonds)`,
          (x) => {
            x.planParams.advanced.returnsStatsForPlanning.expectedValue.empiricalAnnualNonLog =
              {
                type: 'fixed',
                stocks,
                bonds,
              }
          },
        )
      }
    }
  }
  {
    // Stock volatility
    for (const scale of [0.5, 1, 1.5]) {
      add(`stock volatility = ${scale.toFixed(2)}`, (x) => {
        x.planParams.advanced.returnsStatsForPlanning.standardDeviation.stocks.scale =
          { log: scale }
      })
    }
  }
  {
    // Bond volatility
    for (const scale of [0, 1, 2]) {
      add(`bond volatility = ${scale.toFixed(2)}`, (x) => {
        x.planParams.advanced.historicalReturnsAdjustment.standardDeviation.bonds.scale =
          { log: scale }
      })
    }
  }
  {
    // Sampling
    for (const blockSize of [1, 12, 36, 60, 120]) {
      add(`montecarlo block size = ${blockSize}`, (x) => {
        x.planParams.advanced.sampling = {
          type: 'monteCarlo',
          data: {
            blockSize: { inMonths: blockSize },
            staggerRunStarts: true,
          },
        }
      })
    }
    add('historical returns', (x) => {
      x.planParams.advanced.sampling = {
        type: 'historical',
        defaultData: {
          monteCarlo: null,
        },
      }
    })
  }
  return result
}

const _getTests = (
  simulateOnServerResult: SimulateOnServerResult,
  datingInfo: PlanParamsNormalized['datingInfo'],
) => {
  const withdrawal50 =
    simulateOnServerResult.savingsPortfolio.withdrawals.regular
      .byPercentileByMonthsFromNow[1].data
  const minWithdrawals = Math.min(...withdrawal50)
  const maxWithdrawals = Math.max(...withdrawal50)

  const result = [] as { label: string; fn: VariationFn }[]
  for (const marketPerformanceVariation of _getMarketPerformanceVariations()) {
    result.push({
      label: marketPerformanceVariation.label,
      fn: (x) => marketPerformanceVariation.fn(x),
    })
  }

  for (const marketPerformanceVariation of _getMarketPerformanceVariations()) {
    for (const nonCombinatorialTest of _getNonCombinatorialTests(datingInfo)) {
      result.push({
        label: `${marketPerformanceVariation.label} ${nonCombinatorialTest.label}`,
        fn: (x) => {
          marketPerformanceVariation.fn(x)
          nonCombinatorialTest.fn(x)
        },
      })
    }
  }

  for (const timing of _getTimingVariations()) {
    for (const portfolioBalance of _getPortfolioBalanceVariations(
      simulateOnServerResult.portfolioBalanceEstimationByDated.currentBalance,
    )) {
      for (const spendingFloor of _getSpendingFloorVariations(
        minWithdrawals,
        maxWithdrawals,
      )) {
        for (const spendingCeiling of _getSpendingCeilingVariations(
          maxWithdrawals,
          spendingFloor.value,
        )) {
          for (const legacy of _getLegacyVariations()) {
            for (const marketPerformance of _getMarketPerformanceVariations()) {
              for (const strategy of _getStrategyVariations()) {
                if (
                  strategy.label.includes('SWR') &&
                  (spendingFloor.value !== null ||
                    spendingCeiling.value !== null ||
                    legacy.value.total !== 0)
                ) {
                  continue
                }
                if (spendingCeiling.skipOnHighFloor && spendingFloor.isHigh) {
                  continue
                }
                result.push({
                  label: `${timing.label.padEnd(20)} ${portfolioBalance.label.padEnd(25)} ${spendingFloor.label.padEnd(20)} ${spendingCeiling.label.padEnd(20)} ${legacy.label.padEnd(20)} ${marketPerformance.label.padEnd(25)} ${strategy.label}`,

                  fn: (x) => {
                    strategy.fn(x)
                    marketPerformance.fn(x)
                    timing.fn(x)
                    portfolioBalance.fn(x)
                    spendingFloor.fn(x)
                    spendingCeiling.fn(x)
                    legacy.fn(x)
                  },
                })
              }
            }
          }
        }
      }
    }
  }
  // return result.filter(x=>x.label.includes('Legacy=100M') || x.label.includes('Balance=100M'))
  return result
  // return [result[5273]]
}
