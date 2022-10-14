import {paramsInputTypes} from './PlanInputType'

export const planSectionNames = [
  'welcome',
  ...paramsInputTypes,
  'results',
  'summary',
] as const

export type PlanSectionName = typeof planSectionNames[number]
