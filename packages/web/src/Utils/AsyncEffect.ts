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
