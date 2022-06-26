import { noCase } from '../../Utils/Utils'
import { getWASM } from './GetWASM'
import { runSimulationInWASM } from './RunSimulationInWASM'
import { TPAWWorkerArgs, TPAWWorkerResult } from './TPAWWorkerTypes'

addEventListener('message', async event => {
  const eventData: TPAWWorkerArgs = event.data
  const {taskID} = eventData
  switch (eventData.type) {
    case 'runSimulation':
      {
        const result = await runSimulationInWASM(
          eventData.args.params,
          eventData.args.runs
        )
        const reply: TPAWWorkerResult = {type: 'runSimulation', taskID, result}
        ;(postMessage as any)(reply, [
          result.byYearsFromNowByRun.savingsPortfolio.start.balance[0].buffer,
          result.byYearsFromNowByRun.savingsPortfolio.withdrawals.essential[0]
            .buffer,
          result.byYearsFromNowByRun.savingsPortfolio.withdrawals
            .discretionary[0].buffer,
          result.byYearsFromNowByRun.savingsPortfolio.withdrawals.regular[0]
            .buffer,
          result.byYearsFromNowByRun.savingsPortfolio.withdrawals.total[0]
            .buffer,
          result.byYearsFromNowByRun.savingsPortfolio.withdrawals
            .fromSavingsPortfolioRate[0].buffer,
          result.byYearsFromNowByRun.savingsPortfolio.afterWithdrawals
            .allocation.stocks[0].buffer,
          result.byYearsFromNowByRun.savingsPortfolio.excessWithdrawals
            .regular[0].buffer,
          result.byRun.endingBalanceOfSavingsPortfolio.buffer,
        ])
      }
      break
    case 'sort':
      {
        let start = performance.now()
        const {data} = eventData.args

        const wasm = await getWASM()
        const sorted = data.map(row => wasm.sort(row))

        const perf = performance.now() - start
        const reply: TPAWWorkerResult = {
          type: 'sort',
          taskID,
          result: {data: sorted, perf},
        }
        ;(postMessage as any)(
          reply,
          sorted.map(x => x.buffer)
        )
      }
      break
    case 'calculateOneOverCV': {
      let start = performance.now()
      const {data} = eventData.args

      const wasm = await getWASM()
      const oneOverCV = Float64Array.from(
        data.map((row, n) => wasm.one_over_cv(row, n))
      )

      const perf = performance.now() - start
      const reply: TPAWWorkerResult = {
        type: 'calculateOneOverCV',
        taskID,
        result: {data: oneOverCV, perf},
      }
      ;(postMessage as any)(reply), [oneOverCV.buffer]
      break
    }
    case 'clearMemoizedRandom': {
      const wasm = await getWASM()
      wasm.clear_memoized_random()
      const reply: TPAWWorkerResult = {type: 'clearMemoizedRandom', taskID}
      ;(postMessage as any)(reply)
      break
    }
    default:
      noCase(eventData)
  }
})
