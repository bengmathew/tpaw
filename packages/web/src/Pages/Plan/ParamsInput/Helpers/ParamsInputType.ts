import _ from 'lodash'
import { fGet } from '../../../../Utils/Utils'

export const paramsInputTypes = [
  'age-and-retirement',
  'current-portfolio-balance',
  'future-savings',
  'income-during-retirement',
  'extra-spending',
  'legacy',
  'stock-allocation',
  'spending-tilt',
  'spending-ceiling-and-floor',
  'lmp',
  'withdrawal',
  'compare-strategies',
  'expected-returns',
  'inflation',
  'simulation',
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
