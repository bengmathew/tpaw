import { assert, fGet } from '@tpaw/common'
import chalk from 'chalk'
import inquirer from 'inquirer'
import { env } from 'process'
import { Clients } from '../../../Clients.js'
import { cliDevUser } from './CLIDevUser.js'

cliDevUser.command('delete <emailOrId>').action(async (emailOrId: string) => {
  assert(env['NODE_ENV'] === 'development')

  const email = emailOrId.includes('@')
    ? emailOrId
    : fGet((await Clients.firebaseAuth.getUser(emailOrId)).email)

  const { shouldContinue } = (await inquirer.prompt({
    type: 'confirm',
    name: 'shouldContinue',
    message: chalk.red(`Are you sure you want to delete ${email}?`),
    default: false,
  })) as unknown as { shouldContinue: boolean }
  if (!shouldContinue) return
  const firebaseUser = await Clients.firebaseAuth.getUserByEmail(email)

  await Clients.prisma.user.delete({ where: { id: firebaseUser.uid } })
})
