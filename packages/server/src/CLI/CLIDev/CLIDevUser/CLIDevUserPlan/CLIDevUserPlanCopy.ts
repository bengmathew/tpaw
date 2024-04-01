import { fGet, getSlug } from '@tpaw/common'
import fs from 'fs-extra'
import * as uuid from 'uuid'
import { Clients } from '../../../../Clients.js'
import {
  PrismaTransaction,
  serialTransaction,
} from '../../../../Utils/PrismaTransaction.js'
import { cliDevUserPlan } from './CLIDevUserPlan.js'

const cliDevUserPlanCopy = cliDevUserPlan.command('copy')

cliDevUserPlanCopy
  .command('fromDBToFile <srcEmail> <srcPlanId> <destFile>')
  .action(async (srcEmail: string, srcPlanId: string, destFile: string) => {
    const srcUserId = fGet(
      await Clients.firebaseAuth.getUserByEmail(srcEmail),
    ).uid

    const src = await _readFromDB(Clients.prisma, srcUserId, srcPlanId)

    fs.writeJSONSync(destFile, src)
  })

cliDevUserPlanCopy
  .command('fromFileToDB <srcFile> <destEmail> <destLabel>')
  .action(async (srcFile: string, destEmail: string, destLabel: string) => {
    const destUserId = fGet(
      await Clients.firebaseAuth.getUserByEmail(destEmail),
    ).uid
    const src = fs.readJSONSync(srcFile) as _Content

    const destInfo = await serialTransaction(
      async (tx) => await _writeToDB(tx, src, destUserId, destLabel),
    )
    console.dir(destInfo)
  })
  

cliDevUserPlanCopy
  .command('fromDBToDB <srcEmail> <srcPlanId> <destEmail> <destLabel>')
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

      const destInfo = await serialTransaction(
        async (tx) =>
          await _writeToDB(
            tx,
            await _readFromDB(tx, srcUserId, srcPlanId),
            destUserId,
            destLabel,
          ),
      )
      console.dir(destInfo)
    },
  )

type _Content = Awaited<ReturnType<typeof _readFromDB>>
const _readFromDB = async (
  tx: PrismaTransaction,
  userId: string,
  planId: string,
) =>
  await tx.planWithHistory.findUniqueOrThrow({
    where: { userId_planId: { userId, planId } },
    include: { paramsChangeHistory: true },
  })

const _writeToDB = async (
  tx: PrismaTransaction,
  src: _Content,
  userId: string,
  label: string,
) => {
  const planId = uuid.v4()
  const destCurrPlans = await tx.planWithHistory.findMany({
    where: { userId },
  })

  const slug = getSlug(
    label,
    destCurrPlans.map((x) => x.slug),
  )

  await tx.planWithHistory.create({
    data: {
      planId,
      isMain: false,
      userId,
      addedToServerAt: src.addedToServerAt,
      sortTime: src.sortTime,
      lastSyncAt: src.lastSyncAt,
      label,
      slug,
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
  return { slug, planId }
}
