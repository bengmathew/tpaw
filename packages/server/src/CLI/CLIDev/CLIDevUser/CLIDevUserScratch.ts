import { cliDevUser } from './CLIDevUser.js'

cliDevUser
  .command('scratch <emailOrId>')
  .action(async (emailOrId: string) => {})
