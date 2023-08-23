import { assert, fGet } from '@tpaw/common'
import { Clients } from '../../../../Clients.js'
import { cliDevUserPlan } from './CLIDevUserPlan.js'

cliDevUserPlan.command('list <email>').action(async (email, planSlug) => {
  const userId = fGet(await Clients.firebaseAuth.getUserByEmail(email)).uid
  const plans = await Clients.prisma.planWithHistory.findMany({
    where: { userId },
  })
  
  console.dir(plans.map((x) => `${x.label} - (${x.slug})`))
})
