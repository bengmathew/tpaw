import _ from 'lodash'
import { fGet } from '../../../../Utils/Utils'

export const paramsInputTypes = [
  'age-and-retirement',
  'current-portfolio-balance',
  'future-savings',
  'income-during-retirement',
  'extra-spending',
  'spending-ceiling-and-floor',
  'legacy',
  'risk-and-time-preference',
  'strategy',
  'expected-returns',
  'inflation',
  'dev'
] as const
export type ParamsInputType = typeof paramsInputTypes[number]

export const isParamsInputType = (x: string): x is ParamsInputType =>
  (paramsInputTypes as readonly string[]).includes(x)

export const paramsInputOrder = {
  next: (type: ParamsInputType) =>
    type === fGet(_.last(paramsInputTypes))
      ? null
      : paramsInputTypes[paramsInputTypes.indexOf(type) + 1],
  prev: (type: ParamsInputType) =>
    type === paramsInputTypes[0]
      ? null
      : paramsInputTypes[paramsInputTypes.indexOf(type) - 1],
}
