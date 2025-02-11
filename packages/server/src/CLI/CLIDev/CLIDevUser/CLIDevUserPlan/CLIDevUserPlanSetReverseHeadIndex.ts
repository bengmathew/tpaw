import { fGet } from '@tpaw/common'
import { assert } from 'console'
import { Clients } from '../../../../Clients.js'
import { cliDevUserPlan } from './CLIDevUserPlan.js'

cliDevUserPlan
  .command('setReverseHeadIndex <srcEmail> <srcPlanId> <reverseHeadIndex>')
  .action(
    async (
      srcEmail: string,
      srcPlanId: string,
      reverseHeadIndexStr: string,
    ) => {
      const srcUserId = fGet(
        await Clients.firebaseAuth.getUserByEmail(srcEmail),
      ).uid

      const reverseHeadIndex = parseInt(reverseHeadIndexStr)
      assert(Number.isInteger(reverseHeadIndex))

      await Clients.prisma.planWithHistory.update({
        where: {
          userId_planId: {
            userId: srcUserId,
            planId: srcPlanId,
          },
        },
        data: {
          lastSyncAt: new Date(),
          reverseHeadIndex,
        },
      })
    },
  )
