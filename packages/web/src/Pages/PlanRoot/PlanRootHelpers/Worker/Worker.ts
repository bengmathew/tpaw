import {
  PlanParams,
  SomePlanParams,
  assert,
  planParamsMigrate
} from '@tpaw/common'
import { noCase } from '../../../../Utils/Utils'
import {
  WorkerArgs,
  WorkerResult,
} from './WorkerAPI'



// eslint-disable-next-line @typescript-eslint/no-misused-promises
addEventListener('message', async (event) => {
  const eventData: WorkerArgs = event.data
  const { taskID } = eventData
  switch (eventData.type) {
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
    default:
      noCase(eventData.type)
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
    // Sentry.captureException(e)
    assert(e !== null)
    const message =
      // eslint-disable-next-line @typescript-eslint/no-base-to-string
      e instanceof Error ? e.message : `${String(e)} is not Error type.`
    const reply: WorkerResult = {
      type: 'error',
      message,
    }
    ;(postMessage as any)(reply)
  }
}
