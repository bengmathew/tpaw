import { assert, block, noCase } from '@tpaw/common'
import _ from 'lodash'
import pako from 'pako'
import { AppError } from '../../Pages/App/AppError'
import { Config } from '../../Pages/Config'
import { PlanParamsNormalized } from '../NormalizePlanParams/NormalizePlanParams'
import { PortfolioBalanceEstimationCacheHandler } from '../UsePortfolioBalanceEstimationCache'
import { deWire } from './DeWire'
import { getPlanParamsServer } from './GetPlanParamsServer'
import {
  WireSimulationArgs,
  WireSimulationResult,
} from './Wire/wire_simulate_api'

const MAX_RETRIES = 5
const TIMEOUT_MS = 5000

export type SimulateOnServerResult = Awaited<
  ReturnType<typeof simulateOnServer>
>

export type DailyMarketSeriesSrc =
  | { type: 'live' }
  | { type: 'syntheticLiveRepeated' }
  | {
      type: 'syntheticConstant'
      annualPercentageChangeVT: number
      annualPercentageChangeBND: number
    }
export const simulateOnServer = async (
  abortSignal: AbortSignal,
  dailyMarketSeriesSrc: DailyMarketSeriesSrc,
  portfolioBalanceEstimationCacheHandler: PortfolioBalanceEstimationCacheHandler,
  percentiles: number[],
  planParamsNorm: PlanParamsNormalized,
  numOfSimulationsForMonteCarloSampling: number,
  randomSeed: number,
) => {
  const start = performance.now()
  const wireArgs: WireSimulationArgs = {
    marketDailySeriesSrc: block(() => {
      switch (dailyMarketSeriesSrc.type) {
        case 'live':
          return { $case: 'live', live: {} }
        case 'syntheticLiveRepeated':
          return { $case: 'syntheticLiveRepeated', syntheticLiveRepeated: {} }
        case 'syntheticConstant':
          return {
            $case: 'syntheticConstant',
            syntheticConstant: {
              annualPercentageChangeVt:
                dailyMarketSeriesSrc.annualPercentageChangeVT,
              annualPercentageChangeBnd:
                dailyMarketSeriesSrc.annualPercentageChangeBND,
            },
          }
        default:
          noCase(dailyMarketSeriesSrc)
      }
    }),
    timestampForMarketDataMs: planParamsNorm.datingInfo.timestampForMarketData,
    currentPortfolioBalance:
      portfolioBalanceEstimationCacheHandler.wirePortfolioBalanceEstimationArgs,
    percentiles,
    planParams: getPlanParamsServer(
      planParamsNorm,
      numOfSimulationsForMonteCarloSampling,
      randomSeed,
    ),
  }
  console.log('wireArgs', wireArgs)
  const argsEncoded = WireSimulationArgs.encode(wireArgs).finish()
  const argsCompressed = pako.deflate(argsEncoded).buffer as ArrayBuffer

  const authHeaders = _.compact([Config.client.debug.authHeader])
  // Note, fetch does not throw on 400, 500, etc., but will throw on
  // network errors.
  const doFetch = async () =>
    await fetch(`${Config.client.urls.simulator}/2/simulate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Encoding': 'deflate',
        ...(authHeaders ? { authorization: authHeaders.join(', ') } : {}),
      },
      body: argsCompressed,
      signal: AbortSignal.timeout(TIMEOUT_MS),
    })
  let count = 0

  let response: Response | null = null
  const isServerError = (response: Response) =>
    response.status >= 500 && response.status < 600
  while (
    (response === null || isServerError(response)) &&
    count < MAX_RETRIES &&
    !abortSignal.aborted
  ) {
    if (count > 0) await new Promise((resolve) => setTimeout(resolve, 500))
    count++
    try {
      response = await doFetch()
    } catch (e) {
      // So that at the end of the loop the response will reflect the last call.
      response = null
    }
  }
  abortSignal.throwIfAborted()
  if (!response) throw new AppError('networkError')
  if (isServerError(response)) throw new AppError('serverError')

  if (response.headers.get('x-app-error-code') === 'clientNeedsUpdate') {
    throw new AppError('clientNeedsUpdate')
  }

  const dataBlob = await response.blob()
  abortSignal.throwIfAborted()
  const dataArrayBuffer = await dataBlob.arrayBuffer()
  abortSignal.throwIfAborted()

  const wireSimulationResult = WireSimulationResult.decode(
    new Uint8Array(dataArrayBuffer),
  )
  return _processResult(
    wireSimulationResult,
    portfolioBalanceEstimationCacheHandler.handleEstimation,
    planParamsNorm.ages.simulationMonths.numMonths,
    percentiles,
    performance.now() - start,
    argsCompressed.byteLength,
    dataBlob.size,
  )
}

export type NumberArrByPercentileByMonthsFromNow = {
  byPercentileByMonthsFromNow: { data: number[]; percentile: number }[]
}
export type NumberArrWithIdByPercentileByMonthsFromNow = {
  id: string
  byPercentileByMonthsFromNow: { data: number[]; percentile: number }[]
}
const _deWireArrays = (
  wire: AutoDeWiredResult['arrays'],
  numMonths: number,
  percentiles: number[],
  planParamsProcessed: PlanParamsProcessed,
) => {
  const handleByPercentileByMFN = (
    byPercentileByMFNPercentileMajor: number[],
  ): NumberArrByPercentileByMonthsFromNow => {
    return {
      byPercentileByMonthsFromNow: percentiles.map((percentile, i) => ({
        data: byPercentileByMFNPercentileMajor.slice(
          i * numMonths,
          (i + 1) * numMonths,
        ),
        percentile,
      })),
    }
  }

  const handleByPercentile = (
    byPercentile: number[],
  ): { data: number; percentile: number }[] => {
    assert(byPercentile.length === percentiles.length)
    return percentiles.map((percentile, i) => ({
      data: byPercentile[i],
      percentile,
    }))
  }
  return {
    savingsPortfolio: {
      start: {
        balance: handleByPercentileByMFN(
          wire.byPercentileByMfnSimulatedPercentileMajorBalanceStart,
        ),
      },
      withdrawals: {
        essential: block(() => {
          let total = handleByPercentileByMFN(
            wire.byPercentileByMfnSimulatedPercentileMajorWithdrawalsEssential,
          )
          return {
            total,
            byId: _separateExtraWithdrawals(
              planParamsProcessed.amountTimed.adjustmentsToSpending
                .extraSpending.essential,
              total,
            ),
          }
        }),
        discretionary: block(() => {
          let total = handleByPercentileByMFN(
            wire.byPercentileByMfnSimulatedPercentileMajorWithdrawalsDiscretionary,
          )
          return {
            total,
            byId: _separateExtraWithdrawals(
              planParamsProcessed.amountTimed.adjustmentsToSpending
                .extraSpending.discretionary,
              total,
            ),
          }
        }),
        regular: handleByPercentileByMFN(
          wire.byPercentileByMfnSimulatedPercentileMajorWithdrawalsGeneral,
        ),
        total: handleByPercentileByMFN(
          wire.byPercentileByMfnSimulatedPercentileMajorWithdrawalsTotal,
        ),
        fromSavingsPortfolioRate: handleByPercentileByMFN(
          wire.byPercentileByMfnSimulatedPercentileMajorWithdrawalsFromSavingsPortfolioRate,
        ),
      },
      afterWithdrawals: {
        allocation: {
          stocks: handleByPercentileByMFN(
            wire.byPercentileByMfnSimulatedPercentileMajorAfterWithdrawalsAllocationSavingsPortfolio,
          ),
        },
      },
    },
    totalPortfolio: {
      afterWithdrawals: {
        allocationOrZeroIfNoWealth: {
          stocks: handleByPercentileByMFN(
            wire.byPercentileByMfnSimulatedPercentileMajorAfterWithdrawalsAllocationTotalPortfolioOrZeroIfNoWealth,
          ),
        },
      },
    },
    tpawSpendingTilt: {
      total: handleByPercentileByMFN(
        wire.tpawByPercentileByMfnSimulatedPercentileMajorSpendingTilt,
      ),
    },
    endingBalanceOfSavingsPortfolioByPercentile: handleByPercentile(
      wire.byPercentileEndingBalance,
    ),
  }
}

export type AutoDeWiredResult = ReturnType<typeof deWire<WireSimulationResult>>

export type PlanParamsProcessed = ReturnType<typeof _fixPlanParamsProcessed>
const _fixPlanParamsProcessed = (
  src: AutoDeWiredResult['planParamsProcessed'],
) => {
  const _fixAmountTimed = (x: typeof src.amountTimed) => {
    return {
      wealth: {
        futureSavings: x.wealthFutureSavings,
        incomeDuringRetirement: x.wealthIncomeDuringRetirement,
      },
      adjustmentsToSpending: {
        extraSpending: {
          essential: x.extraExpensesEssential,
          discretionary: x.extraExpensesDiscretionary,
        },
      },
    }
  }
  return {
    ...src,
    amountTimed: _fixAmountTimed(src.amountTimed),
  }
}

const _processResult = (
  wireIn: WireSimulationResult,
  handleEstimation: PortfolioBalanceEstimationCacheHandler['handleEstimation'],
  numMonths: number,
  percentiles: number[],
  client_time_in_ms: number,
  compressed_upload_payload_in_bytes: number,
  uncompressed_download_payload_in_bytes: number,
) => {
  const autoDeWired = deWire(wireIn)

  const planParamsProcessed = _fixPlanParamsProcessed(
    autoDeWired.planParamsProcessed,
  )

  const arrays = _deWireArrays(
    autoDeWired.arrays,
    numMonths,
    percentiles,
    planParamsProcessed,
  )

  return {
    portfolioBalanceEstimationByDated: handleEstimation(
      fixPortfolioBalanceEstimationResult(
        autoDeWired.portfolioBalanceEstimationResult,
      ),
    ),
    planParamsProcessed,
    numSimulationsActual: autoDeWired.numRuns,
    numRunsWithInsufficientFunds: autoDeWired.numRunsWithInsufficientFunds,
    ...arrays,
    firstMonthOfSomeRun: _getFirstMonthSavingsPortfolioDetail(
      arrays.savingsPortfolio,
      planParamsProcessed,
    ),
    tpawApproxNetPresentValueForBalanceSheet:
      autoDeWired.tpawNetPresentValueApproxForBalanceSheet,
    performance: {
      client_time_in_ms,
      server_time_in_ms: autoDeWired.performance,
      compressed_upload_payload_in_bytes,
      uncompressed_download_payload_in_bytes,
    },
  }
}

const _separateExtraWithdrawals = (
  processedGroup: PlanParamsProcessed['amountTimed']['wealth']['futureSavings'],
  combinedExtraWithdrawals: Omit<NumberArrByPercentileByMonthsFromNow, 'id'>,
): NumberArrWithIdByPercentileByMonthsFromNow[] =>
  processedGroup.byId.map(({ id, values }) => ({
    id,
    ..._mapByPercentileByMonthsFromNow(
      combinedExtraWithdrawals,
      (value, mfn) => {
        const totalTarget = processedGroup.total[mfn]
        const target = values[mfn]
        if (target === 0) return 0
        const ratio = target / totalTarget
        try {
          assert(!isNaN(ratio)) // if target >0, totalTarget is also >0
        } catch (e) {
          console.log(
            'ratio',
            ratio,
            'target',
            target,
            'totalTarget',
            totalTarget,
          )
          throw e
        }
        return value * ratio
      },
    ),
  }))

const _mapByPercentileByMonthsFromNow = (
  x: NumberArrByPercentileByMonthsFromNow,
  fn: (x: number, i: number) => number,
): NumberArrByPercentileByMonthsFromNow => ({
  byPercentileByMonthsFromNow: x.byPercentileByMonthsFromNow.map((x) => ({
    data: x.data.map(fn),
    percentile: x.percentile,
  })),
})

export type FirstMonthSavingsPortfolioDetail = {
  start: { balance: number }
  contributions: {
    total: number
    toWithdrawal: number
    toSavingsPortfolio: number
  }
  afterContributions: {
    balance: number
  }
  withdrawals: {
    regular: number
    essential: number
    discretionary: number
    total: number
    fromSavingsPortfolio: number
    fromContributions: number
  }
  afterWithdrawals: {
    allocation: { stocks: number }
    balance: number
  }
  contributionToOrWithdrawalFromSavingsPortfolio:
    | { type: 'contribution'; contribution: number }
    | { type: 'withdrawal'; withdrawal: number }
}

const _getFirstMonthSavingsPortfolioDetail = (
  full: ReturnType<typeof _deWireArrays>['savingsPortfolio'],
  planParamsProcessed: PlanParamsProcessed,
): FirstMonthSavingsPortfolioDetail => {
  const _get = (x: NumberArrByPercentileByMonthsFromNow) =>
    x.byPercentileByMonthsFromNow[0].data[0]
  const start = {
    balance: _get(full.start.balance),
  }
  const contributionsTotal =
    planParamsProcessed.amountTimed.wealth.futureSavings.total[0] +
    planParamsProcessed.amountTimed.wealth.incomeDuringRetirement.total[0]
  const afterContributions = {
    balance: start.balance + contributionsTotal,
  }
  const withdrawals = (() => {
    const regular = _get(full.withdrawals.regular)
    const essential = _get(full.withdrawals.essential.total)
    const discretionary = _get(full.withdrawals.discretionary.total)
    const total = _get(full.withdrawals.total)

    const fromContributions = Math.min(total, contributionsTotal)
    const fromSavingsPortfolio = total - fromContributions
    return {
      regular,
      essential,
      discretionary,
      total,
      fromContributions,
      fromSavingsPortfolio,
    }
  })()

  const contributions = {
    total: contributionsTotal,
    toSavingsPortfolio: contributionsTotal - withdrawals.fromContributions,
    toWithdrawal: withdrawals.fromContributions,
  }

  const afterWithdrawals = {
    allocation: {
      stocks: _get(full.afterWithdrawals.allocation.stocks),
    },
    balance: afterContributions.balance - withdrawals.total,
  }
  const contributionToOrWithdrawalFromSavingsPortfolio =
    contributions.toSavingsPortfolio > 0
      ? {
          type: 'contribution' as const,
          contribution: contributions.toSavingsPortfolio,
        }
      : {
          type: 'withdrawal' as const,
          withdrawal: withdrawals.fromSavingsPortfolio,
        }

  return {
    start,
    contributions,
    afterContributions,
    withdrawals,
    afterWithdrawals,
    contributionToOrWithdrawalFromSavingsPortfolio,
  }
}

export namespace PortfolioBalanceEstimation {
  export type Detail = Exclude<
    ReturnType<typeof fixPortfolioBalanceEstimationResult>,
    null
  >
  export type Action = Detail['actions'][number]
}

export const fixPortfolioBalanceEstimationResult = (
  src: AutoDeWiredResult['portfolioBalanceEstimationResult'],
) => {
  if (src === null) return null
  return {
    ...src,
    actions: src.actions.map((x) => ({
      timestamp: x.timestamp,
      args: block(() => {
        switch (x.args.type) {
          case 'marketClose':
            return {
              type: 'marketClose' as const,
              marketData: x.args.marketClose,
            }
          case 'withdrawalAndContribution':
            return {
              type: 'withdrawalAndContribution' as const,
              netContributionOrWithdrawal:
                x.args.withdrawalAndContribution.withdrawalOrContribution,
            }
          case 'monthlyRebalance':
            return {
              type: 'monthlyRebalance' as const,
              allocation: x.args.monthlyRebalance.allocation,
            }
          case 'planChange':
            return {
              type: 'planChange' as const,
              allocation: x.args.planChange.allocation,
              portfolioUpdate: x.args.planChange.portfolioUpdate,
            }
          default:
            noCase(x.args)
        }
      }),
      stateChange: x.stateChange,
    })),
  }
}
