import { faPlus } from '@fortawesome/pro-light-svg-icons'
import {
  faMinus,
  faPlus as faPlusRegular,
  faTrash,
  faTurnDownLeft,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { GlidePath, fGet, generateSmallId } from '@tpaw/common'
import clsx from 'clsx'
import _ from 'lodash'
import React, { useMemo, useState } from 'react'
import {
  NormalizedGlidePath,
  NormalizedGlidePathEntry,
  normalizeGlidePath,
} from '../../../UseSimulator/NormalizePlanParams/NormalizeGlidePath'
import { getMonthToMFN } from '../../../UseSimulator/NormalizePlanParams/NormalizeAges'
import { assert } from '../../../Utils/Utils'
import { useSimulation } from '../../PlanRoot/PlanRootHelpers/WithSimulation'
import { getNormalizedMonthStr } from '../MonthOrDurationDisplay'
import { AmountInput } from './AmountInput'
import { MonthInput } from './MonthInput/MonthInput'
import { CalendarDayFns } from '../../../Utils/CalendarDayFns'

export const GlidePathInput = React.memo(
  ({
    className = '',
    value,
    onChange,
  }: {
    className?: string
    value: NormalizedGlidePath
    onChange: (glidePath: GlidePath) => void
  }) => {
    const { planParamsNorm } = useSimulation()
    const { nowAsCalendarDay } = planParamsNorm.datingInfo
    return (
      <div className={`${className}`}>
        <div className="flex justify-between pb-3 border-b border-gray-500 mb-3">
          <h2 className="font-semibold text-sm">Month</h2>
          <h2 className="font-semibold text-sm">Stock %</h2>
        </div>

        <div
          className="grid gap-y-2 gap-x-2 items-center "
          style={{ grid: 'auto/1fr auto auto' }}
        >
          <h2 className="">Now</h2>
          <_Percent
            className=""
            value={value.now.stocks}
            onChange={(stocks) => {
              const clone = normalizeGlidePath.inverse(value, nowAsCalendarDay)
              clone.start.stocks = stocks
              onChange(clone)
            }}
            modalLabel="Stock Allocation Now"
          />
          <h2 className=""></h2>
          <_Intermediate
            values={value.intermediate}
            onChange={(intermediate) => {
              const clone = normalizeGlidePath.inverse(value, nowAsCalendarDay)
              clone.intermediate = _.fromPairs(
                [
                  ...intermediate,
                  ...value.atOrPastEnd.map(_deNormalizeEntry),
                ].map((x) => [x.id, x]),
              )
              onChange(clone)
            }}
          />
          <h2 className="">Max age</h2>
          <_Percent
            className=""
            value={value.end.stocks}
            onChange={(stocks) => {
              const clone = normalizeGlidePath.inverse(value, nowAsCalendarDay)
              clone.end.stocks = stocks
              onChange(clone)
            }}
            modalLabel="Stock Allocation at Max Age"
          />

          <h2 className=""></h2>
          <_Intermediate
            values={value.atOrPastEnd}
            onChange={(atOrPastEnd) => {
              const clone = normalizeGlidePath.inverse(value, nowAsCalendarDay)
              clone.intermediate = _.fromPairs(
                [
                  ...atOrPastEnd,
                  ...value.intermediate.map(_deNormalizeEntry),
                ].map((x) => [x.id, x]),
              )
              onChange(clone)
            }}
          />
        </div>
        <_NewEntry
          onAdd={(x) => {
            const clone = normalizeGlidePath.inverse(value, nowAsCalendarDay)
            clone.intermediate[x.id] = x
            onChange(clone)
          }}
        />
      </div>
    )
  },
)

const _NewEntry = React.memo(
  ({ onAdd }: { onAdd: (x: GlidePath['intermediate']['string']) => void }) => {
    const { planParamsNorm } = useSimulation()
    const { ages } = planParamsNorm
    const { nowAsCalendarDay } = planParamsNorm.datingInfo
    const [dummyGlidePath, setDummyGlidePath] = useState<GlidePath | null>(
      () => null,
    )
    const handleAdd = () => {
      const id = generateSmallId()
      setDummyGlidePath({
        start: {
          month: {
            type: 'now',
            monthOfEntry: nowAsCalendarDay
              ? {
                  isDatedPlan: true,
                  calendarMonth:
                    CalendarDayFns.toCalendarMonth(nowAsCalendarDay),
                }
              : { isDatedPlan: false },
          },
          stocks: 0.5,
        },
        intermediate: {
          [id]: {
            id,
            indexToSortByAdded: 0,
            month:
              ages.person1.currentAgeInfo.inMonths <
              ages.person1.maxAge.baseValue.inMonths - 1
                ? {
                    type: 'numericAge',
                    person: 'person1',
                    age: {
                      inMonths: ages.person1.currentAgeInfo.inMonths + 1,
                    },
                  }
                : {
                    type: 'numericAge',
                    person: 'person2',
                    age: {
                      inMonths: fGet(ages.person2).currentAgeInfo.inMonths + 1,
                    },
                  },
            stocks: 0.5,
          },
        },
        end: { stocks: 0.5 },
      })
    }

    const normalizedEntry = useMemo((): NormalizedGlidePathEntry | null => {
      if (!dummyGlidePath) return null
      const { intermediate, atOrPastEnd } = normalizeGlidePath(
        dummyGlidePath,
        getMonthToMFN(
          planParamsNorm.datingInfo.nowAsCalendarDay,
          planParamsNorm.ages,
        ),
        planParamsNorm.ages,
        planParamsNorm.datingInfo.nowAsCalendarDay,
      )
      const norm = fGet(_.first([...intermediate, ...atOrPastEnd]))
      // Don't display errors during entry creation.
      return { ...norm, month: { ...norm.month, errorMsg: null } }
    }, [dummyGlidePath, planParamsNorm])

    return dummyGlidePath ? (
      <div className="border-t border-gray-300 rounded-x pt-4 mt-4">
        <div
          className="grid gap-y-2 gap-x-2 items-center"
          style={{ grid: 'auto/1fr auto auto' }}
        >
          <_EntryInput
            normValue={fGet(normalizedEntry)}
            onChange={(x) =>
              setDummyGlidePath({
                ...dummyGlidePath,
                intermediate: { [x.id]: x },
              })
            }
            onDelete={() => setDummyGlidePath(null)}
            alwaysOpen
          />
        </div>
        <div className="flex justify-end mt-4 gap-x-1">
          <button
            className="btn-dark btn-sm"
            onClick={() => {
              onAdd(fGet(_.first(_.values(dummyGlidePath.intermediate))))
              setDummyGlidePath(null)
            }}
          >
            Add
          </button>
        </div>
      </div>
    ) : (
      <button
        className=" btn2-dark btn2-sm flex justify-center items-center gap-x-1 mt-6 "
        onClick={handleAdd}
      >
        <FontAwesomeIcon className="text-base" icon={faPlus} />
        Add Entry
      </button>
    )
  },
)

const _deNormalizeEntry = (
  n: NormalizedGlidePathEntry,
): GlidePath['intermediate']['string'] => ({
  id: n.id,
  indexToSortByAdded: n.indexToSortByAdded,
  month: n.month.baseValue,
  stocks: n.stocks,
})

const _Intermediate = React.memo(
  ({
    values,
    onChange,
  }: {
    values: NormalizedGlidePathEntry[]
    onChange: (values: GlidePath['intermediate']['string'][]) => void
  }) => {
    return (
      <>
        {values.map((x) => {
          return (
            <React.Fragment key={x.id}>
              <_EntryInput
                normValue={x}
                onChange={(x) => {
                  onChange(
                    values.map((y) =>
                      y.id === x.id ? x : _deNormalizeEntry(y),
                    ),
                  )
                }}
                onDelete={() => {
                  onChange(
                    _.compact(
                      values.map((y) =>
                        y.id === x.id ? null : _deNormalizeEntry(y),
                      ),
                    ),
                  )
                }}
                alwaysOpen={false}
              />
              {x.month.errorMsg && (
                <div className=" col-span-3 text-red-500 pl-4">
                  <FontAwesomeIcon
                    className="fa-rotate-90 mb-1 mr-1"
                    icon={faTurnDownLeft}
                  />{' '}
                  {x.month.errorMsg}
                </div>
              )}
            </React.Fragment>
          )
        })}
      </>
    )
  },
)

// Does not display error message, but text-errorFG for month error.
const _EntryInput = React.memo(
  ({
    normValue,
    onChange,
    onDelete,
    alwaysOpen,
  }: {
    normValue: NormalizedGlidePathEntry
    onChange: (value: GlidePath['intermediate'][string]) => void
    onDelete: (() => void) | null
    alwaysOpen: boolean
  }) => {
    const [open, setOpen] = useState(alwaysOpen)
    return (
      <>
        {!open ? (
          <div
            className={clsx(
              normValue.month.errorMsg && ' text-errorFG',
              'text-start',
            )}
          >
            {getNormalizedMonthStr(normValue.month).full}
            <button
              className="text-sm px-2 py-0.5 bg-gray-200 rounded-full ml-2"
              onClick={() => setOpen(true)}
            >
              edit
            </button>
          </div>
        ) : (
          <div
            className={clsx(
              !alwaysOpen && 'bg-gray-100 rounded-xl p-2',
              normValue.month.errorMsg && ' text-errorFG',
            )}
          >
            <MonthInput
              normValue={{
                ...normValue.month,
                // Error message is displayed outside _EntryInput, so don't duplicate it.
                errorMsg: null,
              }}
              onChange={(month) => {
                assert(!('numMonths' in month))
                onChange({
                  id: normValue.id,
                  month,
                  indexToSortByAdded: normValue.indexToSortByAdded,
                  stocks: normValue.stocks,
                })
              }}
              choicesPreFilter={['retirement', 'numericAge', 'calendarMonth']}
              modalTextInputOnMobile
            />
            {!alwaysOpen && (
              <div className="flex justify-end mt-2">
                <button
                  className="text-sm px-2 py-0.5 rounded-full btn-dark"
                  onClick={() => setOpen(false)}
                >
                  Done
                </button>
              </div>
            )}
          </div>
        )}
        <_Percent
          className={clsx(normValue.month.errorMsg && 'text-errorFG')}
          value={normValue.stocks}
          onChange={(stocks) =>
            onChange({ ..._deNormalizeEntry(normValue), stocks })
          }
          modalLabel={`Stock Allocation`}
        />
        {onDelete && (
          <button
            className={clsx(
              normValue.month.errorMsg && 'text-errorFG',
              'px-2 py-1.5 -mr-2',
            )}
            onClick={onDelete}
          >
            <FontAwesomeIcon icon={faTrash} />
          </button>
        )}
      </>
    )
  },
)

const _Percent = React.memo(
  ({
    className = '',
    value,
    onChange,
    modalLabel,
  }: {
    className?: string
    value: number
    onChange: (x: number) => void
    modalLabel: string
  }) => {
    // Rounding is necessary because /100 can result in floating point imprecision.
    const handleChange = (x: number) => onChange(_.round(_.clamp(x, 0, 1), 2))

    return (
      <div className={`${className} flex justify-end items-stretch`}>
        <AmountInput
          className="w-[45px] text-right text-input"
          value={Math.round(value * 100)}
          onChange={handleChange}
          decimals={0}
          modalLabel={modalLabel}
        />
        <button
          className={`flex items-center ml-2 px-2 `}
          onClick={() => handleChange(value + 0.01)}
        >
          <FontAwesomeIcon className="text-base" icon={faPlusRegular} />
        </button>
        <button
          className={`flex items-center px-2 `}
          onClick={() => handleChange(value - 0.01)}
        >
          <FontAwesomeIcon className="text-base" icon={faMinus} />
        </button>
      </div>
    )
  },
)
