import { assert, fGet, getSlug } from '@tpaw/common'
import { Clients } from '../../../../Clients.js'
import { serialTransaction } from '../../../../Utils/PrismaTransaction.js'
import { cliDevUserPlan } from './CLIDevUserPlan.js'
import * as uuid from 'uuid'

cliDevUserPlan
  .command('copy <srcEmail> <srcPlanId> <destEmail> <destLabel>')
  .action(
    async (
      srcEmail: string,
      srcPlanId: string,
      destEmail: string,
      destLabel: string,
    ) => {
      const srcUserId = fGet(
        await Clients.firebaseAuth.getUserByEmail(srcEmail),
      ).uid
      const destUserId = fGet(
        await Clients.firebaseAuth.getUserByEmail(destEmail),
      ).uid

      const destPlanId = uuid.v4()

      const { destSlug } = await serialTransaction(async (tx) => {
        const destCurrPlans = await tx.planWithHistory.findMany({
          where: { userId: destUserId },
        })

        const src = await tx.planWithHistory.findUniqueOrThrow({
          where: { userId_planId: { userId: srcUserId, planId: srcPlanId } },
          include: { paramsChangeHistory: true },
        })

        const destSlug = getSlug(
          destLabel,
          destCurrPlans.map((x) => x.slug),
        )
        await tx.planWithHistory.create({
          data: {
            planId: destPlanId,
            isMain: false,
            userId: destUserId,
            addedToServerAt: src.addedToServerAt,
            sortTime: src.sortTime,
            lastSyncAt: src.lastSyncAt,
            label: destLabel,
            slug: destSlug,
            resetCount: src.resetCount,

            endingParams: fGet(src.endingParams),
            paramsChangeHistory: {
              createMany: {
                data: src.paramsChangeHistory.map((x) => ({
                  // Intentionally not generating a new uuid here, because we
                  // want to  ensure that the planParamsChangeId referred to
                  // other change items (eg. is
                  // currentPortfolioBalance.updatedAtId) are still valid.
                  planParamsChangeId: x.planParamsChangeId,
                  timestamp: x.timestamp,
                  reverseDiff: fGet(x.reverseDiff),
                  change: fGet(x.change),
                })),
              },
            },
            reverseHeadIndex: src.reverseHeadIndex,
          },
        })
        return { destSlug }
      })
      console.log(`Dest slug: ${destSlug}`)
    },
  )
