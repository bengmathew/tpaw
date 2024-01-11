import chalk from 'chalk'
import 'source-map-support/register.js'
import { cli } from './CLI/CLI.js'
import './CLI/CLIScratch.js'
import './HTTP/serve.js'


async function main() {
  try {
    await cli.parseAsync()
    console.log(chalk.green('DONE'))
  } catch (e) {
    console.error(e)
    process.exit(1)
  }
}
await main()
