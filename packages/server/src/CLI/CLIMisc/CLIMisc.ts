import { cli } from '../CLI.js'

export const noop = () => {
  console.dir('abc')
}
cli.command('abc').action(() => console.dir('abc'))
export const cliMisc = cli.command('misc')
