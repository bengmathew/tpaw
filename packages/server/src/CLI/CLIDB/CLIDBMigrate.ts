import { Prisma } from '@prisma/client'
import {
  SomeNonPlanParams,
  SomePlanParams,
  fGet,
  getDefaultNonPlanParams,
  getDefaultPlanParams,
  getSlug,
  planParamsMigrate,
} from '@tpaw/common'
import { DateTime } from 'luxon'
import ora from 'ora'
import * as uuid from 'uuid'
import { Clients } from '../../Clients.js'
import { patchPlanParams } from '../../gql/GQLUser/GQLUserPlan/PatchPlanParams.js'
import { cliDB } from './CLIDB.js'

cliDB.command('migrate').action(async () => {
  const users = await Clients.prisma.user.findMany({
    include: { plan: true },
  })
  console.log(`Num of users: ${users.length}`)
  for (const [index, user] of users.entries()) {
    const { id: userId } = user
    const spinner = ora({
      prefixText: `${index.toString().padStart(4)} User: ${userId}`,
    }).start()
    const now = Date.now()
    if (user.plan) {
      spinner.prefixText += ' - With Plan'
      await Clients.prisma.user.update({
        where: { id: userId },
        data: getUpdateData(now, {
          plan: user.plan.params as SomePlanParams,
          nonPlan: user.plan.params as SomeNonPlanParams,
        }),
      })
      spinner.succeed()
    } else {
      spinner.prefixText += ' - No Plan'
      await Clients.prisma.user.update({
        where: { id: userId },
        data: getUpdateData(now, null),
      })
      spinner.succeed()
    }
  }
})

const getUpdateData = (
  now: number,
  params: { plan: SomePlanParams; nonPlan: SomeNonPlanParams } | null,
): Prisma.UserUpdateInput => {
  const label = null
  const planParams =
    params?.plan ?? getDefaultPlanParams(now, fGet(DateTime.local().zoneName))
  const nonPlanParams = params?.nonPlan ?? getDefaultNonPlanParams()
  return {
    planWithHistory: {
      create: {
        planId: uuid.v4(),
        addedToServerAt: new Date(planParamsMigrate(planParams).timestamp),
        sortTime: new Date(now),
        lastSyncAt: new Date(now),
        label,
        slug: getSlug(label, []),
        isMain: true,
        resetCount: 0,
        endingParams: planParams,
        paramsChangeHistory: {
          create: patchPlanParams.generate({ type: 'forCreate' }, [
            {
              id: uuid.v4(),
              params: planParams,
              change: {
                type: params ? 'startCopiedFromBeforeHistory' : 'start',
                value: null,
              },
            },
          ]),
        },
        reverseHeadIndex: 0,
      },
    },
    nonPlanParams,
    nonPlanParamsLastUpdatedAt: new Date(now),
    clientIANATimezoneName: fGet(DateTime.local().zoneName),
  }
}
