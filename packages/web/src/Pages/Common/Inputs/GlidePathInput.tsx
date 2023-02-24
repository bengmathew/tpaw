import { faPlus } from '@fortawesome/pro-light-svg-icons'
import {
  faMinus,
  faPlus as faPlusRegular,
  faTurnDownLeft,
} from '@fortawesome/pro-regular-svg-icons'
import { faTrash } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { GlidePath, Month } from '@tpaw/common'
import _ from 'lodash'
import React, { useMemo, useState } from 'react'
import { PlanParamsExt } from '../../../TPAWSimulator/PlanParamsExt'
import { numMonthsStr } from '../../../Utils/NumMonthsStr'
import { assert, noCase } from '../../../Utils/Utils'
import { useSimulation } from '../../App/WithSimulation'
import { AmountInput } from './AmountInput'
import { MonthInput } from './MonthInput/MonthInput'

export const GlidePathInput = React.memo(
  ({
    className = '',
    value,
    onChange,
  }: {
    className?: string
    value: GlidePath
    onChange: (glidePath: GlidePath) => void
  }) => {
    const { params, paramsExt } = useSimulation()
    const [newEntry, setNewEntry] = useState<null | _MonthAndStock>(null)

    const intermediate = useMemo(
      () => paramsExt.glidePathIntermediateValidated(value.intermediate),
      [paramsExt, value],
    )

    const handleIntermediateChanged = (
      intermediate: GlidePath['intermediate'],
    ) => {
      const clone = _.cloneDeep(value)
      clone.intermediate = intermediate
      onChange(clone)
    }

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
          <_Intermediate
            value={intermediate}
            onChange={handleIntermediateChanged}
            filter={(x) => x === 'before'}
          />
          <h2 className="">Now</h2>
          <_Percent
            className=""
            value={value.start.stocks}
            onChange={(stocks) => {
              const clone = _.cloneDeep(value)
              clone.start.stocks = stocks
              onChange(clone)
            }}
            modalLabel="Stock Allocation Now"
          />
          <h2 className=""></h2>
          <_Intermediate
            value={intermediate}
            onChange={handleIntermediateChanged}
            filter={(x) => x !== 'before' && x !== 'after'}
          />
          <h2 className="">At max age</h2>
          <_Percent
            className=""
            value={value.end.stocks}
            onChange={(stocks) => {
              const clone = _.cloneDeep(value)
              clone.end.stocks = stocks
              onChange(clone)
            }}
            modalLabel="Stock Allocation at Max Age"
          />
          <_Intermediate
            value={intermediate}
            onChange={handleIntermediateChanged}
            filter={(x) => x === 'after'}
          />

          <h2 className=""></h2>
        </div>
        {newEntry ? (
          <div className="border-t border-gray-300 rounded-x pt-4 mt-4">
            <div
              className="grid gap-y-2 gap-x-2 items-center"
              style={{ grid: 'auto/1fr auto auto' }}
            >
              <_MonthAndStocksInput
                value={newEntry}
                onChange={(x) => setNewEntry(x)}
                onDelete={() => setNewEntry(null)}
                alwaysOpen
              />
            </div>
            <div className="flex justify-end mt-4 gap-x-1">
              <button
                className="btn-dark btn-sm"
                onClick={(x) => {
                  onChange({
                    ...value,
                    intermediate: [...value.intermediate, newEntry],
                  })
                  setNewEntry(null)
                }}
              >
                Add
              </button>
            </div>
          </div>
        ) : (
          <button
            className=" mt-4 flex justify-center items-center gap-x-2  rounded-full "
            onClick={() =>
              setNewEntry({
                month: {
                  type: 'numericAge',
                  person: 'person1',
                  ageInMonths: params.people.person1.ages.currentMonth + 1,
                },
                stocks: 0.5,
              })
            }
          >
            <FontAwesomeIcon className="text-xl" icon={faPlus} />
            Add entry
          </button>
        )}
      </div>
    )
  },
)

type _ProcessedIntermediate = ReturnType<
  PlanParamsExt['glidePathIntermediateValidated']
>

const _Intermediate = React.memo(
  ({
    value,
    onChange,
    filter,
  }: {
    value: _ProcessedIntermediate
    onChange: (value: GlidePath['intermediate']) => void
    filter: (issue: _ProcessedIntermediate[number]['issue']) => boolean
  }) => {
    return (
      <>
        {value.map(
          (x, i) =>
            filter(x.issue) && (
              <React.Fragment key={i}>
                <_MonthAndStocksInput
                  value={x}
                  onChange={(x) => {
                    const result = value.map(({ month, stocks }) => ({
                      month,
                      stocks,
                    }))
                    result[i] = { month: x.month, stocks: x.stocks }
                    onChange(result)
                  }}
                  onDelete={() => {
                    const result = value.map(({ month, stocks }) => ({
                      month,
                      stocks,
                    }))
                    result.splice(i, 1)
                    onChange(result)
                  }}
                  alwaysOpen={false}
                />
                {x.issue !== 'none' && (
                  <div className=" col-span-3 text-red-500">
                    <FontAwesomeIcon
                      className="fa-rotate-90"
                      icon={faTurnDownLeft}
                    />{' '}
                    {x.issue === 'before'
                      ? `Ignoring – out of range.`
                      : x.issue === 'duplicate'
                      ? `Ignoring – duplicate entry for this month.`
                      : x.issue === 'after'
                      ? 'Ignoring – out of range.'
                      : noCase(x.issue)}
                  </div>
                )}
              </React.Fragment>
            ),
        )}
      </>
    )
  },
)

type _MonthAndStock = GlidePath['intermediate'][number]

const _MonthAndStocksInput = React.memo(
  ({
    value,
    onChange,
    onDelete,
    alwaysOpen,
  }: {
    value: _MonthAndStock
    onChange: (value: _MonthAndStock) => void
    onDelete: (() => void) | null
    alwaysOpen: boolean
  }) => {
    const [open, setOpen] = useState(alwaysOpen)
    const { paramsExt } = useSimulation()
    const { asMFN, months, maxMaxAge } = paramsExt

    return (
      <>
        {!open ? (
          <div className="text-start">
            {monthToStringForGlidePath(value.month, paramsExt).full}
            <button
              className="text-sm px-2 py-0.5 bg-gray-200 rounded-full ml-2"
              onClick={() => setOpen(true)}
            >
              edit
            </button>
          </div>
        ) : (
          <div className={`${alwaysOpen ? '' : 'bg-gray-100 rounded-xl p-2'}`}>
            <MonthInput
              location={'standalone'}
              value={value.month}
              onChange={(month) => {
                assert(!('numMonths' in month))
                onChange({ month, stocks: value.stocks })
              }}
              toMFN={(x) => {
                assert(!('numMonths' in x))
                return asMFN(x)
              }}
              range={{
                start: asMFN(months.now) + 1,
                end: asMFN(maxMaxAge) - 1,
              }}
              choices={['retirement', 'numericAge']}
              modalTextInputOnMobile
              getMonthLabel={(value) => {
                assert('type' in value)
                return monthToStringForGlidePath(value, paramsExt).start
              }}
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
          className=""
          value={value.stocks}
          onChange={(stocks) => onChange({ ...value, stocks })}
          modalLabel={`Stock Allocation`}
        />
        {onDelete && (
          <button className="px-2 py-1.5 -mr-2" onClick={onDelete}>
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
    return (
      <div className={`${className} flex justify-end items-stretch`}>
        <AmountInput
          className="w-[45px] text-right text-input"
          value={Math.round(value * 100)}
          onChange={(stocks) => {
            onChange(_.clamp(stocks / 100, 0, 1))
          }}
          decimals={0}
          modalLabel={modalLabel}
        />
        <button
          className={`flex items-center ml-2 px-2 `}
          onClick={() => onChange(_.clamp(value + 0.01, 0, 1))}
        >
          <FontAwesomeIcon className="text-base" icon={faPlusRegular} />
        </button>
        <button
          className={`flex items-center px-2 `}
          onClick={() => onChange(_.clamp(value - 0.01, 0, 1))}
        >
          <FontAwesomeIcon className="text-base" icon={faMinus} />
        </button>
      </div>
    )
  },
)

export const monthToStringForGlidePath = (
  month: Month,
  paramsExt: PlanParamsExt,
) => {
  const { params } = paramsExt

  const result = (startIn: string, date: string | null) => {
    const start = _.capitalize(startIn)
    return {
      start,
      date,
      full: date === null ? start : `${start} ${date}`,
    }
  }
  switch (month.type) {
    case 'now':
      return result('now', null)
    case 'numericAge':
      return result(
        `when ${month.person === 'person1' ? 'you are' : 'your partner is'}`,
        numMonthsStr(month.ageInMonths),
      )
    case 'namedAge': {
      const withPerson = (x: string) =>
        params.people.withPartner
          ? `${month.person === 'person1' ? 'your' : `your partner's`} ${x}`
          : x

      switch (month.age) {
        case 'lastWorkingMonth':
          return result(withPerson('last working month'), null)
        case 'retirement':
          return result(`at ${withPerson('retirement')}`, null)
        case 'max':
          return result(`at ${withPerson('max age')}`, null)
        default:
          noCase(month.age)
      }
    }
    default:
      noCase(month)
  }
}
