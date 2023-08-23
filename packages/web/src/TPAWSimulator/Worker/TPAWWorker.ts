import {
  PlanParams,
  SomePlanParams,
  fGet,
  planParamsMigrate,
} from '@tpaw/common'
import { CurrentPortfolioBalance } from '../../Pages/PlanRoot/PlanRootHelpers/CurrentPortfolioBalance'
import { noCase } from '../../Utils/Utils'
import { getWASM } from './GetWASM'
import { runSimulationInWASM } from './RunSimulationInWASM'
import { TPAWWorkerArgs, TPAWWorkerResult } from './TPAWWorkerAPI'

import * as Sentry from '@sentry/nextjs'

Sentry.init({
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
  const eventData: TPAWWorkerArgs = event.data
  const { taskID } = eventData
  switch (eventData.type) {
    case 'runSimulation':
      {
        const result = runSimulationInWASM(
          eventData.args.params,
          eventData.args.runs,
          await getWASM(),
        )
        const reply: TPAWWorkerResult = {
          type: 'runSimulation',
          taskID,
          result,
        }
        ;(postMessage as any)(reply, [
          result.byMonthsFromNowByRun.savingsPortfolio.start.balance[0].buffer,
          result.byMonthsFromNowByRun.savingsPortfolio.withdrawals.essential[0]
            .buffer,
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
          result.byMonthsFromNowByRun.totalPortfolio.afterWithdrawals.allocation
            .stocks[0].buffer,
          result.byRun.numInsufficientFundMonths.buffer,
          result.byRun.endingBalanceOfSavingsPortfolio.buffer,
        ])
      }
      break
    case 'sort':
      {
        let start = performance.now()
        const { data } = eventData.args

        const wasm = await getWASM()
        const sorted = data.map((row) => wasm.sort(row))

        const perf = performance.now() - start
        const reply: TPAWWorkerResult = {
          type: 'sort',
          taskID,
          result: { data: sorted, perf },
        }
        ;(postMessage as any)(
          reply,
          sorted.map((x) => x.buffer),
        )
      }
      break
    case 'getSampledReturnStats': {
      let start = performance.now()
      const { monthlyReturns, blockSize, numMonths } = eventData.args

      const wasm = await getWASM()
      const { one_year, five_year, ten_year, thirty_year } =
        wasm.get_sampled_returns_stats(
          Float64Array.from(monthlyReturns),
          blockSize,
          numMonths,
        )

      const processYear = ({ n, mean, of_log }: typeof one_year) => ({
        n,
        mean,
        ofLog: {
          n: of_log.n,
          mean: of_log.mean,
          variance: of_log.variance,
          standardDeviation: of_log.standard_deviation,
        },
      })

      const perf = performance.now() - start
      const reply: TPAWWorkerResult = {
        type: 'getSampledReturnStats',
        taskID,
        result: {
          oneYear: processYear(one_year),
          fiveYear: processYear(five_year),
          tenYear: processYear(ten_year),
          thirtyYear: processYear(thirty_year),
          perf,
        },
      }
      ;(postMessage as any)(reply)
      break
    }

    case 'parseAndMigratePlanParams': {
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
      const reply: TPAWWorkerResult = {
        type: 'parseAndMigratePlanParams',
        taskID,
        result: planParamsHistory,
      }
      ;(postMessage as any)(reply)
      break
    }

    case 'estimateCurrentPortfolioBalance': {
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
      const reply: TPAWWorkerResult = {
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
      }
      ;(postMessage as any)(reply)
      break
    }

    case 'clearMemoizedRandom': {
      const wasm = await getWASM()
      wasm.clear_memoized_random()
      const reply: TPAWWorkerResult = { type: 'clearMemoizedRandom', taskID }
      ;(postMessage as any)(reply)
      break
    }
    default:
      noCase(eventData)
  }
})

const planParamsCache = new Map<string, PlanParams>()
