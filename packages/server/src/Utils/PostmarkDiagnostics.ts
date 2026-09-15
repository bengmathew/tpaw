import type { AxiosError, AxiosInstance } from 'axios'
import { AsyncLocalStorage } from 'node:async_hooks'
import type { ServerClient } from 'postmark'

type DiagnosticLog = (
  event: string,
  details?: Record<string, unknown>,
  severity?: 'INFO' | 'ERROR',
) => void

// Keep concurrent sign-in attempts separate while using the shared client.
export const postmarkDiagnosticLog = new AsyncLocalStorage<DiagnosticLog>()

export function addPostmarkDiagnostics(client: ServerClient) {
  const httpClient = client.httpClient.client as AxiosInstance
  // Postmark 4.x otherwise discards Axios's code, cause, and nested errors.
  httpClient.interceptors.response.use(undefined, (error: unknown) => {
    const log = postmarkDiagnosticLog.getStore()
    if (log) {
      const httpError = error as AxiosError<{
        ErrorCode?: number
        Message?: string
      }> | null
      log(
        'postmark.http.failed',
        {
          error: describeEmailError(error),
          httpStatus: httpError?.response?.status ?? null,
          postmarkErrorCode: httpError?.response?.data?.ErrorCode,
          postmarkMessage: httpError?.response?.data?.Message,
          requestTimeoutMs: httpError?.config?.timeout,
        },
        'ERROR',
      )
    }
    return Promise.reject(error)
  })
}

export function describeEmailError(
  error: unknown,
  depth = 0,
): Record<string, unknown> {
  if (depth >= 4) return { truncated: true }
  if (typeof error !== 'object' || error === null) {
    return { message: String(error) }
  }
  const source = error as Record<string, unknown>
  const result: Record<string, unknown> = {}
  // Never serialize request/response bodies, headers, or Axios config: they
  // contain the API token, recipient address, and sign-in link.
  for (const key of [
    'name',
    'message',
    'stack',
    'code',
    'statusCode',
    'errno',
    'syscall',
    'hostname',
    'address',
    'port',
  ]) {
    const value = source[key]
    if (typeof value === 'string' || typeof value === 'number') {
      result[key] = value
    }
  }
  if (source['cause'] !== undefined) {
    result['cause'] = describeEmailError(source['cause'], depth + 1)
  }
  if (Array.isArray(source['errors'])) {
    result['errors'] = source['errors']
      .slice(0, 10)
      .map((nested: unknown) => describeEmailError(nested, depth + 1))
  }
  return result

}