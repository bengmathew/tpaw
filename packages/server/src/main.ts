import 'source-map-support/register.js'
import chalk from 'chalk'
import { cli } from './CLI/CLI.js'
import './CLI/CLIMisc/CLIMisc.js'
import './CLI/CLIDB/CLIDB.js'
import './CLI/CLIScratch.js'

import { Clients } from './Clients.js'
import './gql/serve.js'

async function main() {
  try {
    await cli.parseAsync()
    console.log(chalk.green('DONE'))
  } catch (e) {
    console.error(e)
    process.exit(1)
  } finally {
    await Clients.prisma.$disconnect()
  }
}
await main()
