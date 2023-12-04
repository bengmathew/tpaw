import {
  faCalendarAlt,
  faCaretLeft,
  faCaretRight,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import { fGet } from '@tpaw/common'
import clix from 'clsx'
import _ from 'lodash'
import { DateTime } from 'luxon'
import React, { useMemo, useState } from 'react'
import { ContextMenu2 } from '../../../../Common/Modal/ContextMenu2'
import { useIANATimezoneName } from '../../../PlanRootHelpers/WithNonPlanParams'
import {
  SimulationInfoForHistoryMode,
  useSimulation,
} from '../../../PlanRootHelpers/WithSimulation'
import { usePlanColors } from '../../UsePlanColors'

export const PlanMenuSubMenuRewind = React.memo(
  ({
    simulationInfoForHistoryMode,
  }: {
    simulationInfoForHistoryMode: SimulationInfoForHistoryMode
  }) => {
    const { getZonedTime } = useIANATimezoneName()
    const { currentTimestamp } = useSimulation()
    const { setRewindTo } = simulationInfoForHistoryMode
    const planColors = usePlanColors()
    const prevEndOfDay = getZonedTime(currentTimestamp)
      .minus({ days: 1 })
      .endOf('day')
    const prevOk = !useIsBeforeMin(prevEndOfDay, simulationInfoForHistoryMode)

    const nextEndOfDay = getZonedTime(currentTimestamp)
      .plus({ days: 1 })
      .endOf('day')
    const nextOk = !useIsInFuture(nextEndOfDay, simulationInfoForHistoryMode)

    return (
      <div
        className={clix('px-3 py-1.5 rounded-lg')}
        style={{
          backgroundColor: planColors.results.bg,
          color: planColors.results.fg,
        }}
      >
        <div className="flex gap-x-2">
          <h2 className="">
            {getZonedTime(currentTimestamp).toLocaleString(DateTime.DATE_MED)}
          </h2>
        </div>
        <div className="flex justify-between items-stretch  gap-x-4 mt-1 ">
          <div
            className="grid border rounded-lg "
            style={{
              grid: 'auto/1fr 1fr',
              borderColor: planColors.results.darkBG,
            }}
          >
            <button
              className="pl-4 pr-2 flex items-center disabled:lighten-2"
              disabled={!prevOk}
              onClick={() => setRewindTo(prevEndOfDay.toMillis())}
            >
              <FontAwesomeIcon className="text-xl" icon={faCaretLeft} />
            </button>
            <button
              className="pl-2 pr-4 flex items-center  disabled:lighten-2"
              disabled={!nextOk}
              onClick={() => setRewindTo(nextEndOfDay.toMillis())}
            >
              <FontAwesomeIcon className="text-xl" icon={faCaretRight} />
            </button>
          </div>
          <ContextMenu2
            align="right"
            className="border rounded-lg px-2 "
            style={{
              borderColor: planColors.results.darkBG,
            }}
          >
            <div>
              <FontAwesomeIcon className="text-lg" icon={faCalendarAlt} />
            </div>
            {({ close }) => (
              <Menu.Items className="">
                <div className="p-2" onClick={(e) => {}}>
                  <h2 className="font-bold text-xl">Rewind To</h2>
                  <_Calendar
                    className=""
                    simulationInfoForHistoryMode={simulationInfoForHistoryMode}
                    onClose={close}
                  />
                </div>
              </Menu.Items>
            )}
          </ContextMenu2>
        </div>
      </div>
    )
  },
)

export const _Calendar = React.memo(
  ({
    className,
    simulationInfoForHistoryMode,
    onClose,
  }: {
    className?: string
    simulationInfoForHistoryMode: SimulationInfoForHistoryMode
    onClose: () => void
  }) => {
    const { currentTimestamp } = useSimulation()
    const { getZonedTime } = useIANATimezoneName()
    const { planParamsHistory, actualCurrentTimestamp } =
      simulationInfoForHistoryMode

    const minMonth = getZonedTime(
      fGet(_.first(planParamsHistory)).params.timestamp,
    ).startOf('month')
    const maxMonth = getZonedTime(actualCurrentTimestamp).startOf('month')
    const [month, setMonth] = useState(
      getZonedTime(currentTimestamp).startOf('month'),
    )
    const weeks = useMemo(() => {
      const weeks = [[]] as (DateTime | null)[][]
      _.times(fGet(month.daysInMonth))
        .map((x) => month.plus({ days: x }).endOf('day'))
        .forEach((day, i) => {
          if (day.weekdayShort === 'Sun' && i !== 0) weeks.push([])
          fGet(_.last(weeks)).push(day)
        })
      const n = weeks.length
      weeks[0] = [..._.times(7 - weeks[0].length, () => null), ...weeks[0]]
      weeks[n - 1] = [
        ...weeks[n - 1],
        ..._.times(7 - weeks[n - 1].length, () => null),
      ]
      return weeks
    }, [month])
    return (
      <div className={clix(className, '')}>
        <div className="my-2">
          <div className="flex items-center">
            <button
              className="px-4 py-2 text-xl disabled:lighten-2"
              disabled={
                month.startOf('year').toMillis() ===
                minMonth.startOf('year').toMillis()
              }
              onClick={() =>
                setMonth(
                  DateTime.max(
                    minMonth,
                    month.minus({ years: 1 }).startOf('month'),
                  ),
                )
              }
            >
              <FontAwesomeIcon icon={faCaretLeft} />
            </button>
            <h2 className="text-lg font-semibold w-[100px] text-center">
              {month.year}
            </h2>
            <button
              className="px-4 py-2 text-xl disabled:lighten-2"
              disabled={month.toMillis() === maxMonth.toMillis()}
              onClick={() =>
                setMonth(
                  DateTime.min(
                    maxMonth,
                    month.plus({ years: 1 }).startOf('month'),
                  ),
                )
              }
            >
              <FontAwesomeIcon icon={faCaretRight} />
            </button>
          </div>
          <div className="flex justify-between">
            <div className="flex items-center">
              <button
                className="px-4 py-2 text-xl disabled:lighten-2"
                disabled={month.toMillis() === minMonth.toMillis()}
                onClick={() =>
                  setMonth(month.minus({ months: 1 }).startOf('month'))
                }
              >
                <FontAwesomeIcon icon={faCaretLeft} />
              </button>
              <h2 className="text-lg font-semibold w-[100px] text-center">
                {month.monthLong}
              </h2>
              <button
                className="px-4 py-2 text-xl disabled:lighten-2"
                disabled={month.toMillis() === maxMonth.toMillis()}
                onClick={() =>
                  setMonth(month.plus({ months: 1 }).startOf('month'))
                }
              >
                <FontAwesomeIcon icon={faCaretRight} />
              </button>
            </div>
          </div>
        </div>
        <div
          className="inline-grid text-center  bg-gray-100 rounded-lg  "
          style={{ grid: 'auto/1fr 1fr 1fr 1fr 1fr 1fr 1fr' }}
        >
          {['S', 'M', 'T', 'W', 'T', 'F', 'S'].map((x, i) => (
            <div
              key={i}
              className={clix(
                'w-[45px] h-[45px]  border-b border-gray-600 rounded-m flex items-center justify-center font-medium',
                x === 'S' && 'text-red-500',
              )}
            >
              {x}
            </div>
          ))}
          {weeks.map((week, i) =>
            week.map((endOfDay, j) =>
              endOfDay ? (
                <_CalendarDay
                  key={`${i}-${j}`}
                  endOfDay={endOfDay}
                  simulationInfoForHistoryMode={simulationInfoForHistoryMode}
                  onClose={onClose}
                />
              ) : (
                <div key={`${i}-${j}`}></div>
              ),
            ),
          )}
          {_.range(weeks.length, 6).map((week) => (
            <div key={week} className="h-[45px] my-0.5" />
          ))}
        </div>
        <h2 className="mt-4 flex items-center gap-x-2 text-sm lighten">
          <div className="border border-gray-400 rounded-full w-[20px] h-[20px]"></div>
          days plan was updated
        </h2>
      </div>
    )
  },
)

const _CalendarDay = React.memo(
  ({
    className,
    endOfDay,
    simulationInfoForHistoryMode,
    onClose,
  }: {
    className?: string
    endOfDay: DateTime
    simulationInfoForHistoryMode: SimulationInfoForHistoryMode
    onClose: () => void
  }) => {
    const { currentTimestamp } = useSimulation()
    const { getZonedTime } = useIANATimezoneName()
    const {
      setRewindTo,
      actualCurrentTimestamp,
      planParamsHistory,
      planParamsHistoryEndOfDays,
    } = simulationInfoForHistoryMode
    const isBeforeMin = useIsBeforeMin(endOfDay, simulationInfoForHistoryMode)
    const isInFuture = useIsInFuture(endOfDay, simulationInfoForHistoryMode)
    const isSelected =
      endOfDay.toMillis() ===
      getZonedTime(currentTimestamp).endOf('day').toMillis()
    const [isHover, setIsHover] = useState(false)

    const disabled = isBeforeMin || isInFuture || isSelected
    const isUpdateDay = planParamsHistoryEndOfDays.has(endOfDay.toMillis())

    return (
      <button
        className=""
        disabled={disabled}
        onClick={() => {
          setRewindTo(endOfDay.toMillis())
          // Slight delay to allow the current date selection to be visible
          // before hiding the popup.
          window.setTimeout(() => onClose(), 300)
        }}
        onMouseEnter={() => setIsHover(true)}
        onMouseLeave={() => setIsHover(false)}
      >
        <div
          className={clix(
            className,
            'w-[45px] h-[45px]   rounded-md flex items-center justify-center relative',
            isSelected
              ? 'bg-gray-600 text-white'
              : isHover
              ? 'bg-gray-300'
              : '',
          )}
          // From: https://www.magicpattern.design/tools/css-backgrounds
          style={{
            ...(isBeforeMin || isInFuture
              ? {
                  opacity: '0.2',
                  // backgroundSize: '8px 8px',
                  // backgroundImage: `repeating-linear-gradient(45deg, ${gray[300]} 0, #a7a7b3 0.8px, #f6f6f6 0, #f6f6f6 50%)`,
                }
              : {}),
          }}
        >
          <div
            className={clix(
              'w-[27px] h-[27px] flex items-center justify-center ',
              isUpdateDay && 'rounded-full border',
              isSelected ? 'border-gray-300' : 'border-gray-500',
            )}
          >
            {endOfDay.day}
          </div>
        </div>
      </button>
    )
  },
)

const useIsBeforeMin = (
  endOfDay: DateTime,
  simulationInfoForHistoryMode: SimulationInfoForHistoryMode,
) => {
  const { planParamsHistory } = simulationInfoForHistoryMode
  const { getZonedTime } = useIANATimezoneName()
  return (
    endOfDay.toMillis() <
    getZonedTime(fGet(_.first(planParamsHistory)).params.timestamp)
      .endOf('day')
      .toMillis()
  )
}

const useIsInFuture = (
  endOfDay: DateTime,
  simulationInfoForHistoryMode: SimulationInfoForHistoryMode,
) => {
  const { actualCurrentTimestamp } = simulationInfoForHistoryMode
  const { getZonedTime } = useIANATimezoneName()
  return (
    endOfDay.toMillis() >
    getZonedTime(actualCurrentTimestamp).endOf('day').toMillis()
  )
}
