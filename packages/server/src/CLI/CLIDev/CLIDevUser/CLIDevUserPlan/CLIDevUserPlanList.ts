import { assert, fGet } from '@tpaw/common'
import { Clients } from '../../../../Clients.js'
import { cliDevUserPlan } from './CLIDevUserPlan.js'
import { table } from 'table'

cliDevUserPlan
  .command('list <email>')
  .action(async (email: string, planSlug: string) => {
    const userId = fGet(await Clients.firebaseAuth.getUserByEmail(email)).uid
    const plans = await Clients.prisma.planWithHistory.findMany({
      where: { userId },
    })

    const headers = ['Label', 'Slug', 'Id']
    console.log(
      table([headers, ...plans.map((x) => [x.label, x.slug, x.planId])]),
    )
  })
