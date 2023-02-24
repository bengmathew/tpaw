import { smartDeltaFn } from '../../../Utils/SmartDeltaFn'

export const smartDeltaFnForAmountInput = smartDeltaFn([
  { value: 10000, delta: 1000 },
  { value: 30000, delta: 2000 },
  { value: 100000, delta: 5000 },
  { value: 200000, delta: 10000 },
  { value: 500000, delta: 20000 },
  { value: 1000000, delta: 50000 },
])

export const smartDeltaFnForMonthlyAmountInput = smartDeltaFn([
  { value: 3000, delta: 100 },
  { value: 5000, delta: 200 },
  { value: 10000, delta: 500 },
  { value: 20000, delta: 1000 },
  { value: 50000, delta: 2000 },
  { value: 100000, delta: 5000 },
])
