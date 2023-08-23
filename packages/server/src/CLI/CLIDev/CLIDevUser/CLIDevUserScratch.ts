import { performance } from 'perf_hooks'
import { Clients } from '../../../Clients.js'
import { fGet } from '../../../Utils/Utils.js'
import { cliDevUser } from './CLIDevUser.js'

cliDevUser.command('scratch <emailOrId>').action(async (emailOrId) => {
  const email = emailOrId.includes('@')
    ? emailOrId
    : fGet((await Clients.firebaseAuth.getUser(emailOrId)).email)
  const userId = (await Clients.firebaseAuth.getUserByEmail(email)).uid

  const planSlug = {
    n200000: 'test-plan-200000-100-x-2000',
    n100000: 'test-plan-100000-100-x-1000',
    n10000: 'test-plan-10000-100-x-100',
  }

  // const user = await Clients.prisma.user.findUniqueOrThrow({
  //   where: { id: userId },
  //   include: { planWithHistory :true},
  // })
  // console.dir(user.planWithHistory.map(x=>x.slug))
  const start = performance.now()
  const plan = await Clients.prisma.planWithHistory.findUniqueOrThrow({
    where: { userId_slug: { userId, slug: planSlug.n200000 } },
    include: {
      // paramsChangeHistory:true,
      paramsChangeHistory: {
        orderBy: { timestamp: 'asc' },
        // select: { planParamsChangeId: true},
      },
    },
  })
  const end = performance.now()
  console.log(`Time: ${end - start}ms`)

  console.dir(plan.paramsChangeHistory.length)
})
