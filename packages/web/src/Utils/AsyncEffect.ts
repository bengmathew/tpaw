export function asyncEffect(
  fn: (status: {canceled: boolean}) => Promise<void>
) {
  const status = {canceled: false}
  void fn(status)
  return () => {
    status.canceled = true
  }
}
