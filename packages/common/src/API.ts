import {
  bounded,
  chain,
  failure,
  json,
  JSONGuard,
  object,
  string,
  success,
} from 'json-guard'
import { planParamsGuard } from './PlanParams/Params'

export namespace API {
  const trimmed: JSONGuard<string, string> = (x) =>
    x.trim().length === x.length
      ? success(x)
      : failure('String is not trimmed.')

  const nonEmpty: JSONGuard<string, string> = (x) =>
    x.length > 0 ? success(x) : failure('Empty string.')

  const email: JSONGuard<string> = chain(string, trimmed, (x) => {
    const EMAIL_REGEX = /^[^@]+@([^@]+\.[^@]+)$/
    const DNS_REGEX =
      /^(([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9])\.)*([A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]*[A-Za-z0-9])$/

    const emailMatch = EMAIL_REGEX.exec(x)
    if (emailMatch === null || !emailMatch[1])
      return failure('Email is invalid.')
    if (!DNS_REGEX.test(emailMatch[1]))
      return failure('DNS part of email is invalid')
    return success(x)
  })

  const userId = chain(string, bounded(100))

  export namespace SendSignInEmail {
    export const guards = { email, dest: string }
    export const check = object(guards)
  }

  export namespace SetUserPlan {
    export const check = object({
      userId,
      params: chain(string, json, planParamsGuard),
    })
  }

  export namespace CreateLinkBasedPlan {
    export const check = object({
      params: chain(string, json, planParamsGuard),
    })
  }
}
