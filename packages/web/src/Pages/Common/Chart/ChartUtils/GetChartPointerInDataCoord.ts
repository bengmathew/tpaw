import _ from 'lodash'
import { XY } from '../../../../Utils/Geometry'
import { ChartStateDerived } from '../Chart'

export function getChartPointerInDataCoord(
  pointerInPixelsCoord: XY,
  { scale, plotArea }: ChartStateDerived,
) {
  return {
    x: _.clamp(
      Math.round(scale.x.inverse(pointerInPixelsCoord.x)),
      Math.ceil(scale.x.inverse(plotArea.x)),
      Math.floor(scale.x.inverse(plotArea.right)),
    ),
    y: _.clamp(
      Math.round(scale.y.inverse(pointerInPixelsCoord.y)),
      Math.ceil(scale.y.inverse(plotArea.y)),
      Math.floor(scale.y.inverse(plotArea.right)),
    ),
  }
}
