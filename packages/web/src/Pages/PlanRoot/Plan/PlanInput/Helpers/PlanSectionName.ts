import { paramsInputTypes } from './PlanInputType'

export const planSectionNames = [
  ...paramsInputTypes,
  'help',
  'summary',
] as const

export type PlanSectionName = typeof planSectionNames[number]


