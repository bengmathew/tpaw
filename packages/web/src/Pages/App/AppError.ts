export type AppErrorCode = 'invalidParameters'
export class AppError extends Error {
  code: AppErrorCode
  constructor(code: AppErrorCode, message: string) {
    super(message)
    this.code = code
  }
}
