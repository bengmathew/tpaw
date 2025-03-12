export type AppErrorCode =
  | '404'
  | 'concurrentChange'
  | 'networkError'
  | 'serverError'
  | 'serverDownForMaintenance'
  | 'serverDownForUpdate' // FEATURE: Deprecated.
  | 'clientNeedsUpdate'
export class AppError extends Error {
  code: AppErrorCode
  constructor(code: AppErrorCode, message?: string) {
    super(message ?? code)
    this.code = code
  }
}
