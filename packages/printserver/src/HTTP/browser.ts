import puppeteer from 'puppeteer'

// ---- GLOBAL BROWSER INSTANCE ----
export const browser = await puppeteer.launch({
  headless: 'new',
  args: ['--no-sandbox', '--no-zygote'],
})

const _cleanup = () => {
  void browser.close()
}
process.on('exit', _cleanup)
process.on('uncaughtException', (e) => {
  console.dir(e)
  _cleanup()
  process.exit(1)
})
process.on('unhandledRejection', (e) => {
  console.dir(e)
  _cleanup()
  process.exit(1)
})
process.on('SIGINT', _cleanup)
process.on('SIGTERM', _cleanup)
process.on('SIGQUIT', _cleanup)
