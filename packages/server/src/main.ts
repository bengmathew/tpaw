import 'source-map-support/register.js'
import chalk from 'chalk'
import { cli } from './CLI/CLI.js'
import './CLI/CLIMisc/CLIMisc.js'
import './CLI/CLIDB/CLIDB.js'
import './CLI/CLIDev/CLIDev.js'
import './CLI/CLIDev/CLIDevUser/CLIDevUser.js'
import './CLI/CLIMisc/CLIMisc.js'
import './CLI/CLIMisc/CLIMiscEmailUsers.js'
import './CLI/CLIMisc/CLIMiscPushMarketData.js'
import './CLI/CLIDev/CLIDevUser/CLIDevUserDelete.js'
import './CLI/CLIDev/CLIDevUser/CLIDevUserPlan/CLIDevUserPlan.js'
import './CLI/CLIDev/CLIDevUser/CLIDevUserPlan/CLIDevUserPlanCreateTestPlan.js'
import './CLI/CLIDev/CLIDevUser/CLIDevUserPlan/CLIDevUserPlanCopy.js'
import './CLI/CLIDev/CLIDevUser/CLIDevUserPlan/CLIDevUserPlanDelete.js'
import './CLI/CLIDev/CLIDevUser/CLIDevUserPlan/CLIDevUserPlanList.js'
import './CLI/CLIDev/CLIDevUser/CLIDevUserScratch.js'
import './CLI/CLIDev/CLIDevSendEmail.js'
import './CLI/CLIDB/CLIDBMigrate.js'
import './CLI/CLIScratch/CLIScratch.js'

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
