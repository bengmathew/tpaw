import { assert, fGet } from '@tpaw/common'
import { Clients } from '../../../../Clients.js'
import { serialTransaction } from '../../../../Utils/PrismaTransaction.js'
import { cliDevUserPlan } from './CLIDevUserPlan.js'

cliDevUserPlan
  .command('delete <email> <planSlug>')
  .action(async (email:string, planSlug:string) => {
    const userId = fGet(await Clients.firebaseAuth.getUserByEmail(email)).uid
    await serialTransaction(async (tx) => {
      assert(
        !(
          await tx.planWithHistory.findUniqueOrThrow({
            where: { userId_slug: { userId, slug: planSlug } },
          })
        ).isMain,
      )
      await tx.planWithHistory.delete({
        where: { userId_slug: { userId, slug: planSlug } },
      })
    })
  })
