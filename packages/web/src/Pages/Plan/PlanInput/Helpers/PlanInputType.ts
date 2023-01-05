export const paramsInputTypes = [
  'age',
  'current-portfolio-balance',
  'future-savings',
  'income-during-retirement',
  'extra-spending',
  'legacy',
  'risk',
  'spending-ceiling-and-floor',
  'strategy',
  'expected-returns',
  'inflation',
  'simulation',
  'dev',
] as const
export type PlanInputType = typeof paramsInputTypes[number]

export const isPlanInputType = (
  x: string | null | undefined,
): x is PlanInputType =>
  typeof x === 'string' && (paramsInputTypes as readonly string[]).includes(x)
