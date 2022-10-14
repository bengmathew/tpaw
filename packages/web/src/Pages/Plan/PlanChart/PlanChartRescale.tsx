import {faArrowsUpDown} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import React, {useEffect, useState} from 'react'
import {SimpleRange} from '../../../Utils/SimpleRange'
import {fGet} from '../../../Utils/Utils'
import {useChartData} from '../../App/WithChartData'
import {useSimulation} from '../../App/WithSimulation'
import {PlanSectionName} from '../PlanInput/Helpers/PlanSectionName'
import {PlanChartType} from './PlanChartType'

export const PlanChartRescale = React.memo(
  ({
    section,
    type,
    shouldShowLegacy,
    mainYRange,
    setMainYRange,
    legacyYRange,
    setLegacyYRange,
  }: {
    section: PlanSectionName
    type: PlanChartType
    shouldShowLegacy: boolean
    mainYRange: SimpleRange
    setMainYRange: (x: SimpleRange) => void
    legacyYRange: SimpleRange
    setLegacyYRange: (x: SimpleRange) => void
  }) => {
    const {params} = useSimulation()
    const allChartData = useChartData()
    const chartMainData =
      type === 'reward-risk-ratio-comparison'
        ? fGet(allChartData.rewardRiskRatio)
        : fGet(allChartData.byYearsFromNowPercentiles.get(type))
    const chartLegacyData = allChartData.legacy

    const rescaleWarningLevel = _maxRescaleWarningLevel(
      _rescaleWarningLevel(mainYRange, chartMainData.yDisplayRange),
      shouldShowLegacy
        ? _rescaleWarningLevel(legacyYRange, chartLegacyData.xyDisplayRange.y)
        : 0
    )

    const [pingRescale, setPingRescale] = useState(false)

    useEffect(() => setPingRescale(true), [rescaleWarningLevel])
    const handleRescale = () => {
      setMainYRange(chartMainData.yDisplayRange)
      setLegacyYRange(chartLegacyData.xyDisplayRange.y)
    }

    // This is so chart will be scaled when results are first shown.
    useEffect(() => {
      if (params.dialogMode) handleRescale()
      // Additional section dep and ignore handleRescale.
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [section, params.dialogMode])

    return (
      <button
        className={`absolute bg-cardBG px-3 py-1 flex gap-x-2 items-center rounded-full h-full 
  ${rescaleWarningLevel === 2 ? ' text-red-500 ' : ''}`}
        style={{
          visibility: rescaleWarningLevel === 0 ? 'hidden' : 'visible',
          opacity: rescaleWarningLevel === 0 ? '0' : '1',
          transform: rescaleWarningLevel === 0 ? 'scale(.8)' : 'scale(1)',
          transitionProperty: 'transform, opacity, visibility',
          transitionDuration: '300ms',
        }}
        onClick={handleRescale}
      >
        <FontAwesomeIcon className="" icon={faArrowsUpDown} />
        Rescale
        {rescaleWarningLevel === 2 && (
          <>
            <div className="absolute -right-[2px] -top-[0px] w-[10px] h-[10px] bg-red-500 rounded-full"></div>
            {pingRescale && (
              <div
                className="absolute -right-[12px] -top-[10px] w-[30px] h-[30px] bg-red-500 rounded-full  "
                onAnimationEnd={() => setPingRescale(false)}
                style={{animation: 'ping 1s cubic-bezier(0, 0, 0.2, 1) 4'}}
              ></div>
            )}
            {/* Forces tailwind to insert ping keyframes. */}
            <div className="hidden animate-ping" />
          </>
        )}
      </button>
    )
  }
)

const _rescaleWarningLevel = (
  curr: SimpleRange,
  target: SimpleRange
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

const _maxRescaleWarningLevel = (w1: 0 | 1 | 2, w2: 0 | 1 | 2): 0 | 1 | 2 =>
  Math.max(w1, w2) as 0 | 1 | 2
