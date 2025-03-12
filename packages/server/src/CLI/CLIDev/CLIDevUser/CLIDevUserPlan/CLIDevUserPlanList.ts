import { fGet } from '@tpaw/common'
import { table } from 'table'
import { Clients } from '../../../../Clients.js'
import { cliDevUserPlan } from './CLIDevUserPlan.js'

cliDevUserPlan
  .command('list <email>')
  .action(async (email: string, planSlug: string) => {
    const userId = fGet(await Clients.firebaseAuth.getUserByEmail(email)).uid
    const plans = await Clients.prisma.planWithHistory.findMany({
      where: { userId },
      include: {
        _count: {
          select: {
            paramsChangeHistory: true,
          },
        },
      },
    })

    console.log('userId', userId)
    const headers = [
      'Label',
      'Slug',
      'Id',
      'Is Main',
      'Num of Changes',
      'Created At',
    ]
    console.log(
      table([
        headers,
        ...plans.map((x) => [
          x.label,
          x.slug,
          x.planId,
          `${x.isMain}`,
          x._count.paramsChangeHistory,
          x.addedToServerAt,
        ]),
      ]),
    )
  })
