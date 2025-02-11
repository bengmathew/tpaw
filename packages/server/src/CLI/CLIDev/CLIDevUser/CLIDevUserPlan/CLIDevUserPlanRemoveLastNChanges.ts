import { fGet, SomePlanParams } from '@tpaw/common'
import { assert } from 'console'
import { Clients } from '../../../../Clients.js'
import { patchPlanParams } from '../../../../gql/GQLUser/GQLUserPlan/PatchPlanParams.js'
import { cliDevUserPlan } from './CLIDevUserPlan.js'

cliDevUserPlan
  .command('removeLastNChanges <srcEmail> <srcPlanId> <numOfChangesToRemove>')
  .action(
    async (
      srcEmail: string,
      srcPlanId: string,
      numOfChangesToRemoveStr: string,
    ) => {
      const srcUserId = fGet(
        await Clients.firebaseAuth.getUserByEmail(srcEmail),
      ).uid

      const numOfChangesToRemove = parseInt(numOfChangesToRemoveStr)
      assert(Number.isInteger(numOfChangesToRemove))

      const planWithHistory =
        await Clients.prisma.planWithHistory.findUniqueOrThrow({
          where: {
            userId_planId: {
              userId: srcUserId,
              planId: srcPlanId,
            },
          },
          include: {
            paramsChangeHistory: {
              orderBy: { timestamp: 'desc' },
              // +1 because patchPlanParams returns the plan params at the *end*
              // of each change, so to get the ending params before n changes,
              // we need to take n+1 changes.
              take: numOfChangesToRemove + 1,
            },
          },
        })
      assert(
        planWithHistory.paramsChangeHistory.length === numOfChangesToRemove + 1,
      )

      const newEndingParams = fGet(
        patchPlanParams.forSingle(
          planWithHistory.endingParams as SomePlanParams,
          planWithHistory.paramsChangeHistory,
        ),
      ).params

      const newReverseHeadIndex = Math.max(
        0,
        planWithHistory.reverseHeadIndex - numOfChangesToRemove,
      )

      // console.log('ending', planParamsMigrate(newEndingParams).timestamp)
      // console.log('last', planWithH(newEndingParams).timestamp)

      await Clients.prisma.planWithHistory.update({
        where: {
          userId_planId: {
            userId: srcUserId,
            planId: srcPlanId,
          },
        },
        data: {
          lastSyncAt: new Date(),
          reverseHeadIndex: newReverseHeadIndex,
          endingParams: newEndingParams,
          paramsChangeHistory: {
            deleteMany: {
              planParamsChangeId: {
                // slice(0, -1) because we took n+1 and it is in desc order.
                in: planWithHistory.paramsChangeHistory
                  .slice(0, -1)
                  .map((x) => x.planParamsChangeId),
              },
            },
          },
        },
      })
    },
  )
