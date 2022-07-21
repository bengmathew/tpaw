import {faPlus} from '@fortawesome/pro-light-svg-icons'
import {
  faMinus,
  faPlus as faPlusRegular,
  faTurnDownLeft,
} from '@fortawesome/pro-regular-svg-icons'
import {faTrash} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import React, {useMemo, useState} from 'react'
import {GlidePath} from '../../../TPAWSimulator/TPAWParams'
import {TPAWParamsExt} from '../../../TPAWSimulator/TPAWParamsExt'
import {assert, noCase} from '../../../Utils/Utils'
import {useSimulation} from '../../App/WithSimulation'
import {AmountInput} from './AmountInput'
import {YearInput} from './YearInput/YearInput'

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
    const {params, paramsExt} = useSimulation()
    const [newEntry, setNewEntry] = useState<null | _YearAndStock>(null)
    const [key, setKey] = useState(0)

    const intermediate = useMemo(
      () => paramsExt.glidePathIntermediateValidated(value.intermediate),
      [paramsExt, value]
    )

    const handleIntermediateChanged = (
      intermediate: GlidePath['intermediate']
    ) => {
      const clone = _.cloneDeep(value)
      clone.intermediate = intermediate
      onChange(clone)
      setKey(x => x + 1)
    }

    return (
      <div className={`${className}`} key={key}>
        <div className="flex justify-between pb-3 border-b border-gray-500 mb-3">
          <h2 className="font-semibold text-sm">Year</h2>
          <h2 className="font-semibold text-sm">Stock %</h2>
        </div>

        <div
          className="grid gap-y-2 gap-x-2 items-center "
          style={{grid: 'auto/1fr auto auto'}}
        >
          <_Intermediate
            value={intermediate}
            onChange={handleIntermediateChanged}
            filter={x => x === 'before'}
          />
          <h2 className="">Now</h2>
          <_Percent
            className=""
            value={value.start.stocks}
            onChange={stocks => {
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
            filter={x => x !== 'before' && x !== 'after'}
          />
          <h2 className="">Max Age</h2>
          <_Percent
            className=""
            value={value.end.stocks}
            onChange={stocks => {
              const clone = _.cloneDeep(value)
              clone.end.stocks = stocks
              onChange(clone)
            }}
            modalLabel="Stock Allocation at Max Age"
          />
          <_Intermediate
            value={intermediate}
            onChange={handleIntermediateChanged}
            filter={x => x === 'after'}
          />

          <h2 className=""></h2>
        </div>
        {newEntry ? (
          <div className="border-t border-gray-300 rounded-x pt-4 mt-4">
            <div
              className="grid gap-y-2 gap-x-2 "
              style={{grid: 'auto/1fr auto auto'}}
            >
              <_YearAndStocksInput
                value={newEntry}
                onChange={x => setNewEntry(x)}
                onDelete={() => setNewEntry(null)}
              />
            </div>
            <div className="flex justify-end mt-4 gap-x-1">
              <button
                className="btn-dark btn-sm"
                onClick={x => {
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
                year: {
                  type: 'numericAge',
                  person: 'person1',
                  age: params.people.person1.ages.current + 1,
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
  }
)

type _ProcessedIntermediate = ReturnType<
  TPAWParamsExt['glidePathIntermediateValidated']
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
                <_YearAndStocksInput
                  className=""
                  value={x}
                  onChange={x => {
                    const result = value.map(({year, stocks}) => ({
                      year,
                      stocks,
                    }))
                    result[i] = x
                    onChange(result)
                  }}
                  onDelete={() => {
                    const result = value.map(({year, stocks}) => ({
                      year,
                      stocks,
                    }))
                    result.splice(i, 1)
                    onChange(result)
                  }}
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
                      ? `Ignoring – duplicate entry for this year.`
                      : x.issue === 'after'
                      ? 'Ignoring – out of range.'
                      : noCase(x.issue)}
                  </div>
                )}
              </React.Fragment>
            )
        )}
      </>
    )
  }
)

type _YearAndStock = GlidePath['intermediate'][number]

const _YearAndStocksInput = React.memo(
  ({
    className = '',
    value,
    onChange,
    onDelete,
  }: {
    className?: string
    value: _YearAndStock
    onChange: (value: _YearAndStock) => void
    onDelete: (() => void) | null
  }) => {
    const {paramsExt} = useSimulation()
    const {asYFN, years, maxMaxAge, yearToStr} = paramsExt
    return (
      <>
        <div className="">
          <YearInput
            location={'standalone'}
            value={value.year}
            onChange={year => {
              assert(!('numYears' in year))
              onChange({year, stocks: value.stocks})
            }}
            toYFN={x => {
              assert(!('numYears' in x))
              return asYFN(x)
            }}
            range={{start: asYFN(years.now) + 1, end: asYFN(maxMaxAge) - 1}}
            choices={['lastWorkingYear', 'retirement', 'numericAge']}
            modalTextInputOnMobile
          />
        </div>
        <_Percent
          className=""
          value={value.stocks}
          onChange={stocks => onChange({...value, stocks})}
          modalLabel={`Stock Allocation`}
          />
        {onDelete && (
          <button className="px-2 py-1.5 -mr-2" onClick={onDelete}>
            <FontAwesomeIcon icon={faTrash} />
          </button>
        )}
      </>
    )
  }
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
          onChange={stocks => {
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
  }
)
