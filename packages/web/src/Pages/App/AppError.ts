export type AppErrorCode =
  | '404'
  | '413'
  | 'concurrentChange'
  | 'networkError'
  | 'serverDownForMaintenance'
  | 'serverDownForUpdate'
  | 'clientNeedsUpdate'
export class AppError extends Error {
  code: AppErrorCode
  constructor(code: AppErrorCode, message?: string) {
    super(message ?? code)
    this.code = code
  }
}
