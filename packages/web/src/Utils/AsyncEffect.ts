export function asyncEffect(
  fn: (status: { canceled: boolean }) => Promise<void>,
) {
  const status = { canceled: false }
  // Errors will get swallowed. Make sure there is a handler for
  // unhandledrejection.
  void fn(status)
  return () => {
    status.canceled = true
  }
}

export function asyncEffect2(fn: (signal: AbortSignal) => Promise<void>) {
  const abortController = new AbortController()
  void fn(abortController.signal).catch((e) => {
    if (e instanceof Error && e.name === 'AbortError') {
      console.log('ABORTED')
      return
    }
    throw e
  })
  return () => {
    abortController.abort()
  }
}
