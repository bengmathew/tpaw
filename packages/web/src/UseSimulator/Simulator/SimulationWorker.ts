import {
  PlanParams,
  SomePlanParams,
  assert,
  fGet,
  planParamsMigrate,
} from '@tpaw/common'
import { CurrentPortfolioBalance } from '../../Pages/PlanRoot/PlanRootHelpers/CurrentPortfolioBalance'
import { noCase } from '../../Utils/Utils'
import { getWASM } from './GetWASM'
import { runSimulationInWASM } from './RunSimulationInWASM'
import {
  SimulationWorkerArgs,
  SimulationWorkerResult,
} from './SimulationWorkerAPI'

import * as Sentry from '@sentry/nextjs'

Sentry.init({
  maxValueLength: 10000,
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
})

addEventListener('unhandledrejection', (event) => {
  Sentry.captureException(event.reason)
})
addEventListener('onError', (event) => {
  Sentry.captureException(new Error('onError even in worker.'))
})

// eslint-disable-next-line @typescript-eslint/no-misused-promises
addEventListener('message', async (event) => {
  const eventData: SimulationWorkerArgs = event.data
  const { taskID } = eventData
  switch (eventData.type) {
    case 'runSimulation':
      await _withErrorHandling(async () => {
        const result = runSimulationInWASM(
          eventData.args.currentPortfolioBalanceAmount,
          eventData.args.planParamsRust,
          eventData.args.marketData,
          eventData.args.planParamsNorm,
          eventData.args.planParamsProcessed,
          eventData.args.runs,
          eventData.args.randomSeed,
          await getWASM(),
        )
        return {
          reply: {
            type: 'runSimulation',
            taskID,
            result,
          },
          data: [
            result.byMonthsFromNowByRun.savingsPortfolio.start.balance[0]
              .buffer,
            result.byMonthsFromNowByRun.savingsPortfolio.withdrawals
              .essential[0].buffer,
            result.byMonthsFromNowByRun.savingsPortfolio.withdrawals
              .discretionary[0].buffer,
            result.byMonthsFromNowByRun.savingsPortfolio.withdrawals.regular[0]
              .buffer,
            result.byMonthsFromNowByRun.savingsPortfolio.withdrawals.total[0]
              .buffer,
            result.byMonthsFromNowByRun.savingsPortfolio.withdrawals
              .fromSavingsPortfolioRate[0].buffer,
            result.byMonthsFromNowByRun.savingsPortfolio.afterWithdrawals
              .allocation.stocks[0].buffer,
            result.byMonthsFromNowByRun.totalPortfolio.afterWithdrawals
              .allocation.stocks[0].buffer,
            result.byRun.numInsufficientFundMonths.buffer,
            result.byRun.endingBalanceOfSavingsPortfolio.buffer,
          ],
        }
      })
      break
    case 'sort':
      await _withErrorHandling(async () => {
        let start = performance.now()
        const { data } = eventData.args

        const wasm = await getWASM()
        const sorted = data.map((row) => wasm.sort(row))

        const perf = performance.now() - start
        return {
          reply: {
            type: 'sort',
            taskID,
            result: { data: sorted, perf },
          },
          data: sorted.map((x) => x.buffer),
        }
      })
      break

    case 'parseAndMigratePlanParams':
      await _withErrorHandling(async () => {
        const { planParamsHistoryStr } = eventData.args
        const start = performance.now()
        const planParamsHistory = planParamsHistoryStr.map((x) => ({
          id: x.id,
          params: planParamsMigrate(JSON.parse(x.params) as SomePlanParams),
        }))
        planParamsCache.clear()
        planParamsHistory.forEach(({ id, params }) =>
          planParamsCache.set(id, params),
        )
        return {
          reply: {
            type: 'parseAndMigratePlanParams',
            taskID,
            result: planParamsHistory,
          },
        }
      })
      break

    case 'estimateCurrentPortfolioBalance':
      await _withErrorHandling(async () => {
        const wasm = await getWASM()
        const {
          planId,
          isPreBase,
          planParamsHistory: planParamsHistoryIn,
          estimationTimestamp,
          ianaTimezoneName,
          marketData,
        } = eventData.args
        const planParamsHistory = planParamsHistoryIn.map((x) =>
          x.cached ? { id: x.id, params: fGet(planParamsCache.get(x.id)) } : x,
        )
        if (isPreBase) {
          planParamsCache.clear()
          planParamsHistory.forEach(({ id, params }) =>
            planParamsCache.set(id, params),
          )
        }
        return {
          reply: {
            type: 'estimateCurrentPortfolioBalance',
            taskID,
            result: CurrentPortfolioBalance.getByMonthInfo(
              CurrentPortfolioBalance.getInfo(
                planId,
                planParamsHistory,
                estimationTimestamp,
                ianaTimezoneName,
                marketData,
                wasm,
              ),
            ),
          },
        }
      })
      break
    default:
      noCase(eventData)
  }
})

const planParamsCache = new Map<string, PlanParams>()

const _withErrorHandling = async <T>(
  fn: () => Promise<{ reply: T; data?: unknown[] }>,
) => {
  try {
    const x = await fn()
    ;(postMessage as any)(x.reply, x.data)
  } catch (e) {
    console.dir(e)
    Sentry.captureException(e)
    assert(e !== null)
    const message =
      e instanceof Error ? e.message : `${String(e)} is not Error type.`
    const reply: SimulationWorkerResult = {
      type: 'error',
      message,
    }
    ;(postMessage as any)(reply)
  }
}
