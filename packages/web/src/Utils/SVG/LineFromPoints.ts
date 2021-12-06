import { line } from '@visx/shape'
import { fGet } from '../Utils'

const _optLineFromPoints = line<[number, number]>({
  x: ([x]) => x,
  y: ([x, y]) => y,
})

export const lineFromPoints = (points: [number, number][]) =>
  fGet(_optLineFromPoints(points))
