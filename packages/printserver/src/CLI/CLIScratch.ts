import { cli } from './CLI.js'

cli.command('scratch ').action(async () => {
  console.dir('hello')
})
