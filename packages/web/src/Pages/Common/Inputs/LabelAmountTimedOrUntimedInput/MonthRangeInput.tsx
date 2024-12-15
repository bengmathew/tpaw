import { faCaretDown, faCheck } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Listbox } from '@headlessui/react'
import {
  CalendarMonthFns,
  MonthRange,
  Month,
  assertFalse,
  block,
  fGet,
  noCase,
} from '@tpaw/common'
import clsx from 'clsx'
import React from 'react'
import { NormalizedMonthRange } from '../../../../Simulator/NormalizePlanParams/NormalizeLabeledAmountTimedList/NormalizeAmountAndTimingRecurring'
import { useSimulationInfo } from '../../../PlanRoot/PlanRootHelpers/WithSimulation'
import { ContextModal } from '../../Modal/ContextModal'
import { getMonthRangeDurationStr } from '../../MonthRangeDisplay'
import { MonthInput, MonthType } from '../MonthInput/MonthInput'
import { MonthRangeDurationInput } from '../MonthInput/MonthRangeDurationInput'
import { SimpleModalListbox } from '../../Modal/SimpleModalListbox'
import { getFromMFNToNumericAge } from '../../../../Simulator/NormalizePlanParams/NormalizeAges'

export type MonthRangeInputProps = React.ComponentProps<typeof MonthRangeInput>

type _PropsWithoutValue = {
  className?: string
  onChange: (month: MonthRange) => void
  choicesPreFilter: {
    start: MonthType[]
    end: MonthType[]
  }
}

export const MonthRangeInput = React.memo(
  ({
    normValue,
    className,
    ...props
  }: { normValue: NormalizedMonthRange } & _PropsWithoutValue) => {
    const { planParamsNormInstant } = useSimulationInfo()
    const { datingInfo } = planParamsNormInstant
    const mfnToNumericAge = getFromMFNToNumericAge(planParamsNormInstant)

    const getMonth = (mfn: number): Month =>
      datingInfo.isDated
        ? {
            type: 'calendarMonth',
            calendarMonth: CalendarMonthFns.getFromMFN(
              datingInfo.nowAsCalendarDay,
            )(mfn),
          }
        : mfnToNumericAge.auto(mfn)

    const handleTypeChange = (targetType: NormalizedMonthRange['type']) => {
      switch (targetType) {
        case 'startAndEnd':
          switch (normValue.type) {
            case 'startAndEnd':
              return
            case 'startAndDuration':
              // Exact
              props.onChange({
                type: targetType,
                start: normValue.start.baseValue,
                end: getMonth(normValue.duration.asMFN),
              })
              break
            case 'endAndDuration':
              // Exact
              props.onChange({
                type: targetType,
                start: getMonth(normValue.duration.asMFN),
                end: normValue.end.baseValue,
              })
              break
            default:
              noCase(normValue)
          }
          break
        case 'startAndDuration':
          switch (normValue.type) {
            case 'startAndEnd':
              // Not exact: if end is before start (which includes if end is is
              // in the past), then numMonths is 1, otherwise it is exact. We
              // need to do this because numMonths has to be >0 (enforced by
              // planParamsGuard).
              props.onChange({
                type: targetType,
                start: normValue.start.baseValue,
                duration: {
                  inMonths: normValue.end.isInThePast
                    ? 1
                    : Math.max(
                        1,
                        normValue.end.asMFN - normValue.start.asMFN + 1,
                      ),
                },
              })
              break
            case 'startAndDuration':
              return
            case 'endAndDuration':
              // Exact
              props.onChange({
                type: targetType,
                start: getMonth(normValue.duration.asMFN),
                duration: normValue.duration.baseValue,
              })
              break
            default:
              noCase(normValue)
          }
          break
        case 'endAndDuration':
          switch (normValue.type) {
            case 'startAndEnd':
              // Not exact.
              props.onChange(
                normValue.end.isInThePast ||
                  normValue.end.asMFN < normValue.start.asMFN
                  ? {
                      type: targetType,
                      end: normValue.start.baseValue,
                      duration: { inMonths: 1 },
                    }
                  : {
                      type: targetType,
                      end: normValue.end.baseValue,
                      duration: {
                        inMonths:
                          normValue.end.asMFN - normValue.start.asMFN + 1,
                      },
                    },
              )
              break
            case 'startAndDuration':
              // Exact
              props.onChange({
                type: targetType,
                end: getMonth(normValue.duration.asMFN),
                duration: normValue.duration.baseValue,
              })
              break
            case 'endAndDuration':
              return
            default:
              noCase(normValue)
          }
          break
        default:
          noCase(targetType)
      }
    }
    return (
      <div
        className={clsx(className, 'grid gap-x-3')}
        style={{ grid: 'auto/80px 1fr' }}
      >
        <h2 className="py-1.5">Format:</h2>
        <div className="">
          <SimpleModalListbox
            className="py-1.5"
            value={normValue.type}
            choices={
              ['startAndEnd', 'startAndDuration', 'endAndDuration'] as const
            }
            onChange={handleTypeChange}
            getLabel={_typeToStr}
          />
        </div>

        {/* </div> */}
        {block(() => {
          switch (normValue.type) {
            case 'startAndEnd':
              return (
                <_StartAndEndInput className="" value={normValue} {...props} />
              )
            case 'startAndDuration':
              return (
                <_StartAndDurationInput
                  className=""
                  value={normValue}
                  {...props}
                />
              )
            case 'endAndDuration':
              return (
                <_EndAndDurationInput
                  className=""
                  value={normValue}
                  {...props}
                />
              )
            default:
              noCase(normValue)
          }
        })}
      </div>
    )
  },
)

const _typeToStr = (type: NormalizedMonthRange['type']) => {
  switch (type) {
    case 'startAndEnd':
      return 'Start and End'
    case 'startAndDuration':
      return 'Start and Duration'
    case 'endAndDuration':
      return 'End and Duration'
    default:
      noCase(type)
  }
}

const _StartAndEndInput = React.memo(
  ({
    value,
    onChange,
    choicesPreFilter,
  }: {
    value: Extract<NormalizedMonthRange, { type: 'startAndEnd' }>
  } & _PropsWithoutValue) => {
    const durationStr = getMonthRangeDurationStr(value)
    return (
      <>
        <h2 className="py-1.5">Start:</h2>
        <MonthInput
          className=""
          normValue={value.start}
          onChange={(start) =>
            onChange({
              type: 'startAndEnd',
              start,
              end: value.end.isInThePast
                ? { type: 'inThePast' }
                : value.end.baseValue,
            })
          }
          choicesPreFilter={choicesPreFilter.start}
          modalTextInputOnMobile={false}
        />
        <h2 className="py-1.5">End:</h2>
        <MonthInput
          className=""
          normValue={value.end}
          onChange={(end) =>
            onChange({
              type: 'startAndEnd',
              start: value.start.baseValue,
              end,
            })
          }
          choicesPreFilter={choicesPreFilter.end}
          modalTextInputOnMobile={false}
        />
        {durationStr && <h2 className=" col-span-2 mt-2">{durationStr}</h2>}
      </>
    )
  },
)
const _StartAndDurationInput = React.memo(
  ({
    value,
    onChange,
    choicesPreFilter,
  }: {
    value: Extract<NormalizedMonthRange, { type: 'startAndDuration' }>
  } & _PropsWithoutValue) => {
    return (
      <>
        <h2 className="py-1.5">Start: </h2>
        <MonthInput
          className=""
          normValue={value.start}
          onChange={(start) =>
            onChange({
              type: 'startAndDuration',
              start,
              duration: value.duration.baseValue,
            })
          }
          choicesPreFilter={choicesPreFilter.start}
          modalTextInputOnMobile={false}
        />
        <h2 className="py-1.5">Duration:</h2>
        <MonthRangeDurationInput
          className="py-1.5"
          normValue={value.duration}
          onChange={(duration) =>
            onChange({
              type: 'startAndDuration',
              start: value.start.baseValue,
              duration,
            })
          }
        />
      </>
    )
  },
)
const _EndAndDurationInput = React.memo(
  ({
    value,
    onChange,
    choicesPreFilter,
  }: {
    value: Extract<NormalizedMonthRange, { type: 'endAndDuration' }>
  } & _PropsWithoutValue) => {
    return (
      <>
        <h2 className="py-1.5 ">End: </h2>
        <MonthInput
          className=""
          normValue={value.end}
          onChange={(end) =>
            onChange({
              type: 'endAndDuration',
              end,
              duration: value.duration.baseValue,
            })
          }
          choicesPreFilter={choicesPreFilter.end}
          modalTextInputOnMobile={false}
        />
        <h2 className="py-1.5">Duration:</h2>
        <MonthRangeDurationInput
          className="py-1.5"
          normValue={value.duration}
          onChange={(duration) =>
            onChange({
              type: 'endAndDuration',
              end: value.end.baseValue,
              duration,
            })
          }
        />
      </>
    )
  },
)
