export type AppErrorCode = '404' | 'concurrentChange' | 'networkError'
export class AppError extends Error {
  code: AppErrorCode
  constructor(code: AppErrorCode, message?: string) {
    super(message)
    this.code = code
  }
}
