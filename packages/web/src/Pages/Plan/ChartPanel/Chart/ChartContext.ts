import { LinearFn } from '../../../../Utils/LinearFn'
import { Rect } from './ChartUtils'
import { UseTPAWWorkerResultProcessed } from './DistributionCanvas'

export type ChartContext = {
  ctx: CanvasRenderingContext2D
  tpawResult: UseTPAWWorkerResultProcessed
  percentileInfo: {highlightIndexes: number[]; backgroundIndexes: number[]}
  maxY: {target: number; curr: number}
  padding: {left: number; right: number; top: number; bottom: number}
  viewPort: Rect
  plotArea: Rect
  scale: {
    curr: {x: LinearFn; y: LinearFn}
    target: {x: LinearFn; y: LinearFn}
  }
}
