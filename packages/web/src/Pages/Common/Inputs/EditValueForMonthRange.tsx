import { faMinus, faPlus } from '@fortawesome/pro-regular-svg-icons'
import { faCaretDown, faCaretRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  LabeledAmount,
  PlanParams,
  ValueForMonthRange,
  assert,
  fGet,
  noCase,
} from '@tpaw/common'
import _ from 'lodash'
import React, { useImperativeHandle, useRef, useState } from 'react'
import { formatCurrency } from '../../../Utils/FormatCurrency'
import { Padding } from '../../../Utils/Geometry'
import { SimpleRange } from '../../../Utils/SimpleRange'
import { useSimulation } from '../../App/WithSimulation'
import { ConfirmAlert } from '../Modal/ConfirmAlert'
import { MonthRangeDisplay } from '../MonthRangeDisplay'
import { AmountInput } from './AmountInput'
import { MonthRangeInput, MonthRangeInputProps } from './MonthRangeInput'
import { RealOrNominalInput } from './RealOrNominalInput'
import { smartDeltaFnForMonthlyAmountInput } from './SmartDeltaFnForAmountInput'

type _PropsCommon = {
  mode: 'add' | 'edit'
  title: string
  setHideInMain: (visible: boolean) => void
  onDone: () => void
  onBeforeDelete?: (id: number) => void
  transitionOut: (onDone: () => void) => void
  entryId: number
  labelPlaceholder: string
  cardPadding: Padding
}
type _Props =
  | ({
      hasMonthRange: false
      getEntries: (params: PlanParams) => LabeledAmount[]
    } & _PropsCommon)
  | ({
      hasMonthRange: true
      getEntries: (params: PlanParams) => ValueForMonthRange[]
      choices: MonthRangeInputProps['choices']
      validRangeAsMFN: SimpleRange
    } & _PropsCommon)

type _Section = 'label' | 'amount' | 'monthRange' | 'none'
const _sectionOrder = (
  section: Exclude<_Section, 'none'>,
  hasMonthRange: boolean,
) => {
  switch (section) {
    case 'label':
      return { prev: null, next: 'amount' } as const
    case 'amount':
      return {
        prev: 'label',
        next: hasMonthRange ? 'monthRange' : null,
      } as const
    case 'monthRange':
      return { prev: 'amount', next: null } as const
    default:
      noCase(section)
  }
}

export type EditValueForMonthRangeStateful = { closeSections: () => void }

export const EditValueForMonthRange = React.forwardRef(
  (
    {
      mode,
      title,
      onDone,
      transitionOut,
      setHideInMain,
      onBeforeDelete,
      entryId,
      labelPlaceholder,
      cardPadding,
      ...props
    }: _Props,
    forwardedRef: React.ForwardedRef<EditValueForMonthRangeStateful>,
  ) => {
    const { params, setPlanParams } = useSimulation()
    const outerDivRef = useRef<HTMLDivElement>(null)
    const buttonDivRef = useRef<HTMLDivElement>(null)

    const [{ dialogMode, currSection }, setState] = useState({
      dialogMode: mode === 'add',
      currSection: mode === 'add' ? 'label' : ('none' as _Section),
    })
    const [confirmDelete, setConfirmDelete] = useState(false)
    const endDialogMode = () =>
      setState({ dialogMode: false, currSection: 'none' })

    const setCurrSection = (x: _Section) => {
      setState((curr) =>
        curr.dialogMode && x === 'none' ? curr : { ...curr, currSection: x },
      )
    }

    function processGetEntries<T extends { id: number }>(
      getEntries: (params: PlanParams) => T[],
    ) {
      const entry = fGet(getEntries(params.plan).find((x) => x.id === entryId))
      const setEntry = (editEntry: (entry: T) => void) => {
        setPlanParams((plan) => {
          const clone = _.cloneDeep(plan)
          const entryClone = _.cloneDeep(entry)
          editEntry(entryClone)
          const index = getEntries(clone).findIndex((x) => x.id === entryId)
          assert(index !== -1)
          getEntries(clone).splice(index, 1, entryClone)
          return clone
        })
      }
      return {
        entry,
        setEntry,
        currSection,
        cardPadding,
        setCurrSection,
        dialogMode,
        endDialogMode,
      }
    }

    const sectionProps = props.hasMonthRange
      ? {
          hasMonthRange: true as const,
          choices: props.choices,
          validRangeAsMFN: props.validRangeAsMFN,
          ...processGetEntries(props.getEntries),
        }
      : {
          hasMonthRange: false as const,
          ...processGetEntries(props.getEntries),
        }

    useImperativeHandle(
      forwardedRef,
      () => ({
        closeSections: () => setCurrSection('none'),
      }),
      [],
    )

    const handleDelete = () => {
      setHideInMain(true)
      onBeforeDelete?.(sectionProps.entry.id)
      transitionOut(() => {
        onDone()
        setPlanParams((plan) => {
          const clone = _.cloneDeep(plan)
          const entries = props.getEntries(clone)
          const index = entries.findIndex((x) => x.id === entryId)
          // Needed because transitionOut seems to be called multiple times.
          if (index === -1) return plan
          props.getEntries(clone).splice(index, 1)
          return clone
        })
      })
    }

    return (
      <div
        ref={outerDivRef}
        className={``}
        onClick={(e) => {
          if (e.target === outerDivRef.current) setCurrSection('none')
        }}
      >
        <h2
          className="text-lg sm:text-xl font-bold text-center"
          onClick={() => setCurrSection('none')}
        >
          {title}
        </h2>

        <_LabelSection
          className="mt-8"
          {...sectionProps}
          labelPlaceholder={labelPlaceholder}
        />
        <_AmountSection className="mt-8" {...sectionProps} />
        {sectionProps.hasMonthRange && (
          <_MonthRangeSection className="mt-8" {...sectionProps} />
        )}
        <div
          ref={buttonDivRef}
          className={`mt-10 flex gap-x-4 ${
            mode === 'add' ? 'justify-between' : 'justify-end'
          }`}
          onClick={(e) => {
            if (e.target === buttonDivRef?.current) setCurrSection('none')
          }}
        >
          {mode === 'add' ? (
            <button className="btn-md btn-md btn-outline" onClick={handleDelete}>
              Cancel
            </button>
          ) : (
            <button
              className="text-errorFG btn-md"
              onClick={() => setConfirmDelete(true)}
            >
              Delete
            </button>
          )}
          {!dialogMode && (
            <button
              className="btn-dark btn-md"
              onClick={() => {
                setHideInMain(false)
                transitionOut(onDone)
              }}
            >
              Done
            </button>
          )}
        </div>
        {confirmDelete && (
          <ConfirmAlert
            option1={{
              label: 'Delete',
              onClose: () => handleDelete(),
              isWarning: true,
            }}
            onCancel={() => setConfirmDelete(false)}
          >
            Are you sure you want to delete this entry?
          </ConfirmAlert>
        )}
      </div>
    )
  },
)

type _SectionPropsCommon = {
  className?: string
  cardPadding: Padding
  currSection: _Section
  setCurrSection: (x: _Section) => void
  dialogMode: boolean
  endDialogMode: () => void
}

type _SectionProps =
  | ({
      hasMonthRange: true
      entry: ValueForMonthRange
      setEntry: (edit: (x: ValueForMonthRange) => void) => void
    } & _SectionPropsCommon)
  | ({
      hasMonthRange: false
      entry: LabeledAmount
      setEntry: (edit: (x: LabeledAmount) => void) => void
    } & _SectionPropsCommon)

const _LabelSection = React.memo(
  (props: _SectionProps & { labelPlaceholder: string }) => {
    const { entry, setEntry, labelPlaceholder } = props
    return (
      <_Section {...props} sectionType="label" sectionName="Label">
        <>
          {entry.label ? (
            <h2 className="">{entry.label}</h2>
          ) : (
            <h2 className="lighten">none</h2>
          )}
        </>
        <input
          type="text"
          className="bg-gray-200 px-2 py-1.5 rounded-lg w-full max-w-[400px] mt-2"
          value={entry.label ?? ''}
          onChange={(e) => {
            setEntry((entry) => {
              const trimmed = e.target.value.trim()
              // Cannot set it to trimmed value because we cannot have
              entry.label = trimmed.length === 0 ? null : e.target.value
            })
          }}
          placeholder={labelPlaceholder}
        />
      </_Section>
    )
  },
)
const _AmountSection = React.memo((props: _SectionProps) => {
  const { entry, setEntry, hasMonthRange } = props
  const { increment, decrement } = smartDeltaFnForMonthlyAmountInput
  return (
    <_Section
      {...props}
      sectionType="amount"
      sectionName={hasMonthRange ? 'Amount per Month' : 'Amount'}
    >
      <h2 className="">
        {formatCurrency(entry.value)} {entry.nominal ? 'not' : ''} adjusted for
        inflation ({entry.nominal ? 'nominal' : 'real'} dollars)
      </h2>
      <div
        className="grid gap-x-2  mt-2"
        style={{
          grid: 'auto auto /  auto',
        }}
      >
        <div className="flex " >
          <AmountInput
            className="w-[100px] text-input"
            prefix="$"
            value={entry.value}
            onChange={(value) => setEntry((entry) => (entry.value = value))}
            decimals={0}
            modalLabel={null}
          />
          <button
            className="ml-3 px-3"
            onClick={() =>
              setEntry((entry) => (entry.value = increment(entry.value)))
            }
          >
            <FontAwesomeIcon icon={faPlus} />
          </button>
          <button
            className="px-3"
            onClick={() =>
              setEntry((entry) => (entry.value = decrement(entry.value)))
            }
          >
            <FontAwesomeIcon icon={faMinus} />
          </button>
        </div>
        <RealOrNominalInput
          className="mt-4"
          nominal={entry.nominal}
          onChange={(nominal) => setEntry((entry) => (entry.nominal = nominal))}
        />
      </div>
    </_Section>
  )
})

const _MonthRangeSection = React.memo(
  ({
    validRangeAsMFN,
    choices,
    ...props
  }: Extract<_SectionProps, { hasMonthRange: true }> & {
    validRangeAsMFN: SimpleRange
    choices: MonthRangeInputProps['choices']
  }) => {
    const { paramsExt } = useSimulation()
    const { clampMonthRangeToNow, months } = paramsExt
    const { entry, setEntry } = props
    const monthRangeClamped = clampMonthRangeToNow(entry.monthRange)

    return (
      <_Section
        {...props}
        sectionType="monthRange"
        sectionName="For These Months "
      >
        <MonthRangeDisplay
          className=""
          valueClamped={monthRangeClamped}
          range={validRangeAsMFN}
          skipLength={false}
        />
        <>
          {monthRangeClamped ? (
            <MonthRangeInput
              className=""
              valueClamped={monthRangeClamped}
              setValue={(monthRange) =>
                setEntry((entry) => (entry.monthRange = monthRange))
              }
              rangeAsMFN={validRangeAsMFN}
              choices={choices}
              modalTextInputOnMobile={false}
            />
          ) : (
            <div>
              <MonthRangeDisplay
                className=""
                valueClamped={monthRangeClamped}
                range={validRangeAsMFN}
                skipLength={false}
              />
              <button
                className="underline pt-2"
                onClick={() => {
                  setEntry(
                    (entry) =>
                      (entry.monthRange = {
                        type: 'startAndNumMonths',
                        start: months.now,
                        numMonths: 1,
                      }),
                  )
                }}
              >
                Edit dates
              </button>
            </div>
          )}
        </>
      </_Section>
      // ) : (
      // <div className={`${props.className ?? ''}`}>
      //   <h2 className="inline bg-gray-300 px-2 py-0.5 rounded-lg ">Completed</h2>
      //   <h2 className="mt-2">The duration for this entry is in the past.</h2>
      // </div>
    )
  },
)

const _Section = React.memo(
  ({
    className = '',
    sectionType,
    sectionName,
    currSection,
    setCurrSection,
    cardPadding,
    dialogMode,
    endDialogMode,
    hasMonthRange,
    children: [summaryChild, editChild],
  }: {
    sectionType: Exclude<_Section, 'none'>
    sectionName: string
    children: [React.ReactNode, React.ReactNode]
  } & _SectionProps) => {
    const sectionOrder = _sectionOrder(sectionType, hasMonthRange)
    return currSection !== sectionType ? (
      <button
        className={`${className} block text-start`}
        onClick={() => setCurrSection(sectionType)}
      >
        <h2 className="font-bold">
          <FontAwesomeIcon className="w-[10px]" icon={faCaretRight} />{' '}
          {sectionName}
        </h2>
        <div className="ml-3">{summaryChild}</div>
      </button>
    ) : (
      <div className={`${className} `}>
        <div
          className="bg-gray-100 rounded-xl"
          style={{
            paddingBottom: `15px`,
            paddingTop: `10px`,
            marginTop: `-11px`,
            paddingLeft: `${cardPadding.left * 0.25}px`,
            marginLeft: `-${cardPadding.left * 0.25 + 1}px`,
            paddingRight: `${cardPadding.right * 0.25}px`,
            marginRight: `-${cardPadding.right * 0.25 + 1}px`,
          }}
        >
          <button
            className="font-bold mb-2"
            onClick={() => setCurrSection('none')}
          >
            <FontAwesomeIcon className="w-[10px]" icon={faCaretDown} />{' '}
            {sectionName}
          </button>
          <div className="ml-3">{editChild}</div>
          {dialogMode && (
            <div className="flex justify-end mt-4 px-2 gap-x-2">
              {sectionOrder.prev && (
                <button
                  className="btn-sm btn-outline"
                  onClick={() => setCurrSection(sectionOrder.prev)}
                >
                  Prev
                </button>
              )}
              <button
                className="btn-sm btn-dark"
                onClick={() => {
                  if (sectionOrder.next) {
                    setCurrSection(sectionOrder.next)
                  } else {
                    endDialogMode()
                  }
                }}
              >
                Next
              </button>
            </div>
          )}
        </div>
      </div>
    )
  },
)
