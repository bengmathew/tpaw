import { faArrowsUpDown } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import React, { useCallback, useEffect } from 'react'
import { SimpleRange } from '../../../../Utils/SimpleRange'
import { fGet } from '../../../../Utils/Utils'
import { useSimulation } from '../../PlanRootHelpers/WithSimulation'
import { PlanSectionName } from '../PlanInput/Helpers/PlanSectionName'
import { useChartData } from '../WithChartData'
import { PlanResultsChartType } from './PlanResultsChartType'

export const PlanResultsChartRescale = React.memo(
  ({
    section,
    chartType,
    mainYRange,
    setMainYRange,
  }: {
    section: PlanSectionName
    chartType: PlanResultsChartType
    mainYRange: SimpleRange
    setMainYRange: (x: SimpleRange) => void
  }) => {
    const { planParams } = useSimulation()
    const allChartData = useChartData()
    const chartMainData = fGet(
      allChartData.byYearsFromNowPercentiles.get(chartType),
    )

    const rescaleWarningLevel = _rescaleWarningLevel(
      mainYRange,
      chartMainData.yDisplayRange,
    )

    const handleRescale = useCallback(
      () => setMainYRange(chartMainData.yDisplayRange),
      [chartMainData.yDisplayRange, setMainYRange],
    )

    useEffect(() => {
      if (rescaleWarningLevel === 0) return
      const timeout = window.setTimeout(handleRescale, 30 * 1000)
      return () => window.clearTimeout(timeout)
    }, [rescaleWarningLevel, handleRescale])

    // This is so chart will be scaled when results are first shown.
    useEffect(() => {
      if (planParams.dialogPosition === 'show-results') handleRescale()
      // Additional section dep and ignore handleRescale.
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [section, planParams.dialogPosition])

    return (
      <button
        className={`absolute  px-5  flex gap-x-2 items-center rounded-full h-full  
  ${
    rescaleWarningLevel === 2
      ? 'bg-red-600 text-red-100 '
      : 'bg-red-600 text-red-100'
  }`}
        style={{
          visibility: rescaleWarningLevel === 0 ? 'hidden' : 'visible',
          opacity: rescaleWarningLevel === 0 ? '0' : '1',
          transform: rescaleWarningLevel === 0 ? 'scale(.8)' : 'scale(1)',
          transitionProperty: 'transform, opacity, visibility',
          transitionDuration: '300ms',
          boxShadow: '0px 0px 20px 5px rgba(0,0,0,0.38)',
        }}
        onClick={handleRescale}
      >
        <FontAwesomeIcon className="" icon={faArrowsUpDown} />
        Rescale
      </button>
    )
  },
)

const _rescaleWarningLevel = (
  curr: SimpleRange,
  target: SimpleRange,
): 0 | 1 | 2 => {
  const length = Math.abs(target.end - target.start)
  const missing =
    (Math.max(0, curr.start - target.start) +
      Math.max(0, target.end - curr.end)) /
    length
  const excess =
    (Math.max(0, target.start - curr.start) +
      Math.max(0, curr.end - target.end)) /
    length

  if (missing > 0.5 || excess > 1.25) return 2
  if (missing > 0.025 || excess > 0.05) return 1
  return 0
}
