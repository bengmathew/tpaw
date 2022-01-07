import {faTrash} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {Switch} from '@headlessui/react'
import _ from 'lodash'
import React, {useState} from 'react'
import {
  numericYear,
  TPAWParams,
  ValueForYearRange,
  YearRangeEdge,
  YearRangeFixedEdge,
} from '../../../../TPAWSimulator/TPAWParams'
import {fGet, noCase} from '../../../../Utils/Utils'
import {AmountInput, useAmountInputState} from '../../../Common/Inputs/AmountInput'
import {SliderInput} from '../../../Common/Inputs/SliderInput/SliderInput'
import {ToggleSwitch} from '../../../Common/Inputs/ToggleSwitch'
import {ConfirmAlert} from '../../../Common/Modal/ConfirmAlert'
import {ModalBase} from '../../../Common/Modal/ModalBase'

const allPossibleFixes = [
  'start',
  'lastWorkingYear',
  'retirement',
  'end',
] as YearRangeFixedEdge[]

export const ValueForYearRangeInput = React.memo(
  ({
    initial,
    params,
    heading,
    onCancel,
    onDone,
    onDelete,
    type,
  }: {
    initial: Omit<ValueForYearRange, 'value'> & {value: number | null}
    heading: string
    params: TPAWParams
    onCancel: () => void
    onDone: (result: ValueForYearRange) => void
    onDelete: (() => void) | null
    type: 'full' | 'beforeRetirement' | 'afterRetirement'
  }) => {
    const [confirmDelete, setConfirmDelete] = useState(false)
    const [label, setLabel] = useState(initial.label ?? '')
    const amountState = useAmountInputState(initial.value)

    const [domainStart, domainEnd] = (() => {
      const {age} = params
      switch (type) {
        case 'full':
          return [age.start, age.end]
        case 'beforeRetirement':
          return [age.start, age.retirement - 1]
        case 'afterRetirement':
          return [age.retirement, age.end]
        default:
          noCase(type)
      }
    })()

    const [yearRange, setYearRange] = useState(() => {
      const {age} = params
      const numeric = (x: YearRangeEdge) =>
        _.clamp(numericYear({age}, x), domainStart, domainEnd)
      const start = numeric(initial.yearRange.start)
      const end = Math.max(start, numeric(initial.yearRange.end))
      return {start, end}
    })

    const isSingleYear = yearRange.start === yearRange.end
    const [nominal, setnominal] = useState(initial.nominal)
    const [fixes, setFixes] = useState(() => {
      const ends = [initial.yearRange.start, initial.yearRange.end]
      const fixesThatCouldHaveBeenButWereNot = allPossibleFixes.filter(x =>
        ends.includes(numericYear(params, x))
      )
      return _.difference(allPossibleFixes, fixesThatCouldHaveBeenButWereNot)
    })
    const fixesToDisplay = _.compact(
      [yearRange.start, yearRange.end].map(x =>
        x === numericYear(params, 'start')
          ? ('start' as const)
          : x === numericYear(params, 'lastWorkingYear')
          ? ('lastWorkingYear' as const)
          : x === numericYear(params, 'retirement')
          ? ('retirement' as const)
          : x === numericYear(params, 'end')
          ? ('end' as const)
          : undefined
      )
    )
    const setEnabledForFix =
      (fix: YearRangeFixedEdge) =>
      (enabled: boolean | ((enabled: boolean) => boolean)) => {
        setFixes(fixes =>
          (
            typeof enabled === 'boolean'
              ? enabled
              : enabled(fixes.includes(fix))
          )
            ? _.union(fixes, [fix])
            : _.difference(fixes, [fix])
        )
      }

    const isRetired = params.age.start === params.age.retirement
    const domain = _.range(domainStart, domainEnd + 1).map(value => ({
      value,
      tick:
        value % 10 === 0
          ? ('large' as const)
          : value % 1 === 0
          ? ('small' as const)
          : ('none' as const),
    }))
    const sliderProps = {
      className: '',
      formatValue: (x: number) => `${x}`,
      domain,
      height: 60,
    }
    const handleSave = () => {
      const year = (x: number) =>
        x === params.age.start && fixes.includes('start')
          ? 'start'
          : x === params.age.retirement - 1 && fixes.includes('lastWorkingYear')
          ? 'lastWorkingYear'
          : x === params.age.retirement && fixes.includes('retirement')
          ? 'retirement'
          : x === params.age.end && fixes.includes('end')
          ? 'end'
          : x
      const labelTrim = label.trim()
      onDone({
        label: labelTrim.length === 0 ? null : labelTrim,
        value: amountState.amount,
        yearRange: {
          start: year(yearRange.start),
          end: year(yearRange.end),
        },
        nominal,
      })
    }
    const clampYear = (year: number) =>
      _.clamp(year, domainStart, domainEnd)

    const FixSwitch = React.memo(
      ({fix, label}: {fix: YearRangeFixedEdge; label: string}) => (
        <Switch.Group>
          <div className="flex justify-end items-center ">
            <Switch.Label className="mr-4  ">{label}</Switch.Label>
            <ToggleSwitch
              enabled={fixes.includes(fix)}
              setEnabled={setEnabledForFix(fix)}
            />
          </div>
        </Switch.Group>
      )
    )
    return (
      <>
        <ModalBase>
          {transitionOut => (
            <>
              <h2 className="text-lg font-bold text-center">{heading}</h2>
              <div className=" p-2 w-[min(100vw-32px,400px)]">
                {/* Dummy button to capture focus on mobile so keyboard won't show */}
                <button className=""></button>
                <div
                  className="grid gap-x-4 items-center"
                  style={{grid: '50px 50px 50px 35px auto/ auto 1fr'}}
                >
                  <h2 className="justify-self-end">Label</h2>
                  <input
                    type="text"
                    className="bg-gray-200 px-2 py-1.5 rounded-lg"
                    value={label}
                    onChange={e => setLabel(e.target.value)}
                  />
                  <div className="justify-self-end">
                    <h2 className="">Amount</h2>
                    <h2
                      className="text-right lighten-2 text-[.75rem] "
                      style={{lineHeight: '.75rem'}}
                    >
                      per year
                    </h2>
                  </div>
                  <AmountInput className="" state={amountState} />
                  <Switch.Group>
                    <Switch.Label className="justify-self-end">
                      Nominal
                    </Switch.Label>
                    <ToggleSwitch
                      className=""
                      enabled={nominal}
                      setEnabled={setnominal}
                    />
                  </Switch.Group>
                  <Switch.Group>
                    <Switch.Label className="justify-self-end whitespace-nowrap ">
                      Single Year{' '}
                    </Switch.Label>
                    <ToggleSwitch
                      className=""
                      enabled={isSingleYear}
                      setEnabled={isSingleYear => {
                        setYearRange(yearRange => {
                          if (isSingleYear) {
                            return {
                              start: yearRange.start,
                              end: yearRange.start,
                            }
                          } else if (yearRange.start + 1 > params.age.end) {
                            return {
                              start: yearRange.start - 1,
                              end: yearRange.start,
                            }
                          } else {
                            return {
                              start: yearRange.start,
                              end: yearRange.start + 1,
                            }
                          }
                        })
                      }}
                    />
                  </Switch.Group>
                  <h2 className="justify-self-end ">
                    {isSingleYear ? 'Year' : 'Years'}
                  </h2>
                  {isSingleYear ? (
                    <SliderInput
                      key="single"
                      {...sliderProps}
                      pointers={[
                        {value: clampYear(yearRange.start), type: 'normal'},
                      ]}
                      onChange={([start]) => setYearRange({start, end: start})}
                    />
                  ) : (
                    <SliderInput
                      key="double"
                      {...sliderProps}
                      pointers={[
                        {value: clampYear(yearRange.start), type: 'left'},
                        {value: clampYear(yearRange.end), type: 'right'},
                      ]}
                      onChange={([start, end]) => setYearRange({start, end})}
                    />
                  )}
                </div>

                <div className="h-14 overflow-hidden  mt-1 text-sm p-1 grid gap-y-1">
                  {fixesToDisplay.includes('start') && (
                    <FixSwitch label="Fix to Current Year" fix="start" />
                  )}
                  {fixesToDisplay.includes('lastWorkingYear') && (
                    <FixSwitch
                      label="Fix to Last Working Year"
                      fix="lastWorkingYear"
                    />
                  )}
                  {fixesToDisplay.includes('retirement') && (
                    <FixSwitch label="Fix to Retirement" fix="retirement" />
                  )}
                  {fixesToDisplay.includes('end') && (
                    <FixSwitch label="Fix to Max Age" fix="end" />
                  )}
                </div>
                <div
                  className="grid gap-x-2 items-center"
                  style={{grid: 'auto/auto auto 1fr auto'}}
                >
                  <button
                    className="btn-lg btn-dark  "
                    onClick={() => transitionOut(handleSave)}
                  >
                    Save
                  </button>
                  <button
                    className="btn-lg btn-none"
                    onClick={() => transitionOut(onCancel)}
                  >
                    Cancel
                  </button>
                  <div />
                  {onDelete ? (
                    <button
                      className="px-2 -mr-2 py-2 btn-none text-errorFG justify-end text-xl"
                      onClick={() => setConfirmDelete(true)}
                    >
                      <FontAwesomeIcon icon={faTrash} />
                    </button>
                  ) : (
                    <div />
                  )}
                </div>
              </div>
            </>
          )}
        </ModalBase>
        {confirmDelete && (
          <ConfirmAlert
            title="Confirm Delete"
            confirmText="Delete"
            isWarning
            onCancel={() => setConfirmDelete(false)}
            onConfirm={() => fGet(onDelete)()}
          >
            Are you sure you want to delete this entry?
          </ConfirmAlert>
        )}
      </>
    )
  }
)
