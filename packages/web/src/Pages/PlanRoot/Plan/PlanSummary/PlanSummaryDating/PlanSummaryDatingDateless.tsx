import { faCaretDown, faInfinity } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  PLAN_PARAMS_CONSTANTS,
  getLastMarketDataDayForUndatedPlans,
  getNYZonedTime,
} from '@tpaw/common'
import clsx from 'clsx'
import { DateTime } from 'luxon'
import React, { useMemo, useState } from 'react'
import { NormalizedDatingInfo } from '../../../../../Simulator/NormalizePlanParams/NormalizePlanParams'
import { CalendarDayFns } from '../../../../../Utils/CalendarDayFns'
import { CalendarDayInput } from '../../../../Common/Inputs/CalendarDayInput'
import { ContextModal } from '../../../../Common/Modal/ContextModal'
import { useIANATimezoneName } from '../../../PlanRootHelpers/WithNonPlanParams'
import {
  useSimulationInfo,
  useSimulationResultInfo,
} from '../../../PlanRootHelpers/WithSimulation'
import { PlanMenuActionModalConvertDatingLocal } from '../../PlanMenu/PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalConvertDatingLocal'
import { PlanMenuActionModalConvertDatingServer } from '../../PlanMenu/PlanMenuActions/PlanMenuActionModals/PlanMenuActionModalConvertDatingServer'

export const PlanSummaryDatingDateless = React.memo(
  ({
    className,
    datingInfo,
  }: {
    className?: string
    datingInfo: Extract<NormalizedDatingInfo, { isDated: false }>
  }) => {
    const { updatePlanParams, simulationInfoBySrc } = useSimulationInfo()
    const { simulationResult } = useSimulationResultInfo()
    const { getZonedTime } = useIANATimezoneName()
    const [isCalendarOpen, setIsCalendarOpen] = React.useState(false)
    const [showConvertToDated, setShowConvertToDated] = useState(false)
    const { dateStr, valueInfo, range } = useMemo(() => {
      const dateTimeForMarketData = getNYZonedTime.fromObject(
        datingInfo.marketDataAsOfEndOfDayInNY,
      )
      const valueInfo =
        dateTimeForMarketData.startOf('day').toMillis() ===
        dateTimeForMarketData.toMillis()
          ? ({
              hasValue: true,
              value: dateTimeForMarketData,
            } as const)
          : ({
              hasValue: false,
              startingMonth: dateTimeForMarketData.startOf('month'),
            } as const)

      const range = {
        start: getNYZonedTime(PLAN_PARAMS_CONSTANTS.minPlanParamTime).startOf(
          'day',
        ),
        end: getNYZonedTime.fromObject(
          getLastMarketDataDayForUndatedPlans(datingInfo.nowAsTimestampNominal),
        ),
      }
      return {
        dateStr: {
          timestampForMarketData: CalendarDayFns.toStr(
            datingInfo.marketDataAsOfEndOfDayInNY,
          ),
          now: getZonedTime(datingInfo.nowAsTimestampNominal).toLocaleString(
            DateTime.DATE_FULL,
          ),
        },
        valueInfo,
        range,
      }
    }, [datingInfo, getZonedTime])
    return (
      <div className={clsx(className)}>
        {/* <h2 className="text-right">{dateStr.now}</h2> */}
        <p className="p-base">
          <FontAwesomeIcon className="text-base mr-1" icon={faInfinity} /> This
          is a dateless plan. It is not tied to the current date and does not
          change over time. Recommended for examples and not for personal
          planning.{' '}
          <button
            className="underline"
            onClick={() => setShowConvertToDated(true)}
          >
            Convert to dated plan
          </button>
          .
        </p>
        <div className="flex gap-x-2 mt-1 p-base">
          <h2 className="py-1.5">Use market data as of:</h2>
          <div className="">
            <ContextModal
              align="left"
              open={isCalendarOpen}
              onOutsideClickOrEscape={() => setIsCalendarOpen(false)}
            >
              {({ ref }) => (
                <button
                  ref={ref}
                  className="py-1.5"
                  onClick={() => setIsCalendarOpen(true)}
                >
                  {dateStr.timestampForMarketData}
                  <FontAwesomeIcon className="ml-2" icon={faCaretDown} />
                </button>
              )}
              <div className="p-2">
                <CalendarDayInput
                  valueInfo={valueInfo}
                  range_memoized={range}
                  onChange={(day) => {
                    updatePlanParams(
                      'setMarketDataDay',
                      CalendarDayFns.fromDateTime(day),
                    )
                    // Delay so we can see change in calendar before close.
                    window.setTimeout(() => setIsCalendarOpen(false), 300)
                  }}
                  shouldHighlightDay={false}
                />
              </div>
            </ContextModal>
          </div>
        </div>

        {simulationInfoBySrc.src === 'server' ? (
          <PlanMenuActionModalConvertDatingServer
            show={showConvertToDated}
            onHide={() => setShowConvertToDated(false)}
            isSyncing={simulationInfoBySrc.syncState.type !== 'synced'}
            plan={simulationInfoBySrc.plan}
            reload={simulationInfoBySrc.reload}
          />
        ) : (
          <PlanMenuActionModalConvertDatingLocal
            show={showConvertToDated}
            onHide={() => setShowConvertToDated(false)}
            onConvert={simulationInfoBySrc.reset}
            skipNoUndoCopy={simulationInfoBySrc.src === 'link'}
          />
        )}
      </div>
    )
  },
)
