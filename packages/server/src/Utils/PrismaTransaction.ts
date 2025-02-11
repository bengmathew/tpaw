import { Prisma, PrismaClient } from '@prisma/client'
import { setTimeout } from 'timers/promises'
import { Clients } from '../Clients.js'

export type PrismaTransaction = Omit<
  PrismaClient,
  '$connect' | '$disconnect' | '$on' | '$transaction' | '$use' | '$extends'
>

export const serialTransaction = async <T>(
  fn: (tx: PrismaTransaction) => Promise<T>,
  {
    retries = 3,
    onConflict = () => {},
    label,
    timeout, 
  }: { retries?: number; onConflict?: () =>void; label?: string; timeout?: number } = {},
): Promise<T> => {
  try {
    return await Clients.prisma.$transaction(fn, {
      isolationLevel: Prisma.TransactionIsolationLevel.Serializable,
      timeout, // Default is 5000ms
    })
  } catch (e) {
    // P2034 is conflict:
    // https://www.prisma.io/docs/reference/api-reference/error-reference#p2034
    const isConflict =
      e instanceof Prisma.PrismaClientKnownRequestError && e.code === 'P2034'
    if (retries === 0 || !isConflict) throw e
    onConflict()
    await setTimeout(1000)
    return await serialTransaction(fn, { retries: retries - 1 })
  }
}
