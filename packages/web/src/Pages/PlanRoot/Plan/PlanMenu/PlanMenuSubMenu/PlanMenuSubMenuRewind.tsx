import {
  faCalendarAlt,
  faCaretLeft,
  faCaretRight,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import { assert, fGet } from '@tpaw/common'
import clix from 'clsx'
import _ from 'lodash'
import { DateTime } from 'luxon'
import React, { useMemo, useState } from 'react'
import { ContextModal } from '../../../../Common/Modal/ContextModal'
import { useIANATimezoneName } from '../../../PlanRootHelpers/WithNonPlanParams'
import {
  SimulationInfoForHistoryMode,
  SimulationInfoForServerSrc,
  useSimulation,
} from '../../../PlanRootHelpers/WithSimulation'
import { usePlanColors } from '../../UsePlanColors'
import { CalendarDayInput } from '../../../../Common/Inputs/CalendarDayInput'
import { CalendarDayFns } from '../../../../../Utils/CalendarDayFns'

export const PlanMenuSubMenuRewind = React.memo(
  ({
    simulationInfoForServerSrc,
    simulationInfoForHistoryMode,
  }: {
    simulationInfoForServerSrc: SimulationInfoForServerSrc
    simulationInfoForHistoryMode: SimulationInfoForHistoryMode
  }) => {
    const {setRewindTo} = simulationInfoForServerSrc
    const { planParamsNorm } = useSimulation()
    assert(planParamsNorm.datingInfo.isDated)
    const { nowAsTimestamp } = planParamsNorm.datingInfo
    const { getZonedTime } = useIANATimezoneName()
    const {
      planParamsHistory,
      actualCurrentTimestamp,
      planParamsHistoryDays,
    } = simulationInfoForHistoryMode
    const planColors = usePlanColors()

    const range_memoized = useMemo(
      () => ({
        start: getZonedTime(
          fGet(_.first(planParamsHistory)).params.timestamp,
        ).startOf('day'),
        end: getZonedTime(actualCurrentTimestamp).startOf('day'),
      }),
      [getZonedTime, planParamsHistory, actualCurrentTimestamp],
    )
    const { value, prevDay, nextDay } = useMemo(() => {
      const value = getZonedTime(nowAsTimestamp).startOf('day')
      const prevDay = value.minus({ days: 1 }).startOf('day')
      const nextDay = value.plus({ days: 1 }).startOf('day')
      return { value, prevDay, nextDay }
    }, [getZonedTime, nowAsTimestamp])

    const prevOk = prevDay.toMillis() >= range_memoized.start.toMillis()
    const nextOk = nextDay.toMillis() <= range_memoized.end.toMillis()

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
            {getZonedTime(nowAsTimestamp).toLocaleString(DateTime.DATE_MED)}
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
              onClick={() => setRewindTo(CalendarDayFns.fromDateTime(prevDay))}
            >
              <FontAwesomeIcon className="text-xl" icon={faCaretLeft} />
            </button>
            <button
              className="pl-2 pr-4 flex items-center  disabled:lighten-2"
              disabled={!nextOk}
              onClick={() => setRewindTo(CalendarDayFns.fromDateTime(nextDay))}
            >
              <FontAwesomeIcon className="text-xl" icon={faCaretRight} />
            </button>
          </div>
          <Menu>
            {({ open, close }) => (
              <ContextModal
                align="right"
                open={open}
                onOutsideClickOrEscape={null}
              >
                {({ ref }) => (
                  <Menu.Button
                    ref={ref}
                    className="border rounded-lg px-2 "
                    style={{
                      borderColor: planColors.results.darkBG,
                    }}
                  >
                    <FontAwesomeIcon className="text-lg" icon={faCalendarAlt} />
                  </Menu.Button>
                )}
                <Menu.Items className="">
                  <div className="p-2">
                    <h2 className="font-bold text-xl">Rewind To</h2>
                    <CalendarDayInput
                      className=""
                      // TODO: handle no value.
                      valueInfo={{ hasValue: true, value }}
                      range_memoized={range_memoized}
                      shouldHighlightDay={(day) =>
                        planParamsHistoryDays.has(day.toMillis())
                      }
                      onChange={(day) => {
                        setRewindTo(CalendarDayFns.fromDateTime(day))
                        // Slight delay to allow the current date selection to be visible
                        // before hiding the popup.
                        window.setTimeout(() => close(), 300)
                      }}
                    />
                    <h2 className="mt-2 flex items-center gap-x-2 text-sm lighten">
                      <div className="border border-gray-400 rounded-full w-[20px] h-[20px]"></div>
                      days plan was updated
                    </h2>
                  </div>
                </Menu.Items>
              </ContextModal>
            )}
          </Menu>
        </div>
      </div>
    )
  },
)
