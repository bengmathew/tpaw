import { faMinus, faPlus } from '@fortawesome/pro-regular-svg-icons'
import { faCaretDown, faCaretRight } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { MonthRange, fGet, noCase } from '@tpaw/common'
import React, {
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react'
import { formatCurrency } from '../../../Utils/FormatCurrency'
import { Padding } from '../../../Utils/Geometry'
import { SimpleRange } from '../../../Utils/SimpleRange'
import { optGet } from '../../../Utils/optGet'
import {
  getLabeledAmountEntriesByLocation,
  getValueForMonthRangeEntriesByLocation,
} from '../../PlanRoot/PlanRootHelpers/PlanParamsChangeAction'
import { useSimulation } from '../../PlanRoot/PlanRootHelpers/WithSimulation'
import { CenteredModal } from '../Modal/CenteredModal'
import { MonthRangeDisplay } from '../MonthRangeDisplay'
import { AmountInput } from './AmountInput'
import { MonthRangeInput, MonthRangeInputProps } from './MonthRangeInput'
import { RealOrNominalInput } from './RealOrNominalInput'
import { smartDeltaFnForMonthlyAmountInput } from './SmartDeltaFnForAmountInput'

type _PropsCommon = {
  addOrEdit: 'add' | 'edit'
  title: string
  setHideInMain: (visible: boolean) => void
  onDone: () => void
  onBeforeDelete?: (id: string) => void
  transitionOut: (onDone: () => void) => void
  entryId: string
  labelPlaceholder: string
  cardPadding: Padding
}
type _Props =
  | ({
      hasMonthRange: false
      location: Parameters<typeof getLabeledAmountEntriesByLocation>[1]
    } & _PropsCommon)
  | ({
      hasMonthRange: true
      location: Parameters<typeof getValueForMonthRangeEntriesByLocation>[1]
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
      addOrEdit,
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
    const { updatePlanParams } = useSimulation()
    const outerDivRef = useRef<HTMLDivElement>(null)
    const buttonDivRef = useRef<HTMLDivElement>(null)

    const { planParams } = useSimulation()
    const [{ dialogMode, currSection }, setState] = useState({
      dialogMode: addOrEdit === 'add',
      currSection: addOrEdit === 'add' ? 'label' : ('none' as _Section),
    })
    const [confirmDelete, setConfirmDelete] = useState(false)
    const endDialogMode = () =>
      setState({ dialogMode: false, currSection: 'none' })

    const setCurrSection = (x: _Section) => {
      setState((curr) =>
        curr.dialogMode && x === 'none' ? curr : { ...curr, currSection: x },
      )
    }

    const sectionPropsCommon = {
      entryId,
      currSection,
      cardPadding,
      setCurrSection,
      dialogMode,
      endDialogMode,
    }
    const sectionProps = props.hasMonthRange
      ? {
          location: props.location,
          hasMonthRange: true as const,
          choices: props.choices,
          validRangeAsMFN: props.validRangeAsMFN,
          ...sectionPropsCommon,
        }
      : {
          location: props.location,
          hasMonthRange: false as const,
          ...sectionPropsCommon,
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
      onBeforeDelete?.(sectionProps.entryId)
      transitionOut(() => {
        onDone()
        updatePlanParams('deleteLabeledAmount', {
          location: props.location,
          entryId,
        })
      })
    }

    // Handle the case where entry is missing. This can happen through undo/redo.
    const entry = useMemo(
      () =>
        getLabeledAmountEntriesByLocation(planParams, props.location)[entryId],
      [entryId, planParams, props.location],
    )
    const handleNoEntry = () => transitionOut(() => {})
    const handleNoEntryRef = useRef(handleNoEntry)
    handleNoEntryRef.current = handleNoEntry
    useEffect(() => {
      if (!entry) handleNoEntryRef.current()
    }, [entry])

    if (!entry) return <></>
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
            addOrEdit === 'add' ? 'justify-between' : 'justify-end'
          }`}
          onClick={(e) => {
            if (e.target === buttonDivRef?.current) setCurrSection('none')
          }}
        >
          {addOrEdit === 'add' ? (
            <button
              className="btn-md btn-md btn-outline"
              onClick={handleDelete}
            >
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
        <CenteredModal
          className=" dialog-outer-div"
          show={confirmDelete}
          onOutsideClickOrEscape={null}
        >
          <h2 className=" dialog-heading">Confirm Delete</h2>
          <p className=" p-base dialog-content-div">
            Are you sure you want to delete this entry?
          </p>
          <div className=" dialog-button-div">
            <button
              className=" dialog-button-cancel"
              onClick={() => setConfirmDelete(false)}
            >
              Cancel
            </button>
            <button
              className=" dialog-button-warning"
              onClick={() => {
                setConfirmDelete(false)
                handleDelete()
              }}
            >
              Delete
            </button>
          </div>
        </CenteredModal>
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
  entryId: string
}

type _SectionProps =
  | ({
      hasMonthRange: true
      location: Parameters<typeof getValueForMonthRangeEntriesByLocation>[1]
    } & _SectionPropsCommon)
  | ({
      hasMonthRange: false
      location: Parameters<typeof getLabeledAmountEntriesByLocation>[1]
    } & _SectionPropsCommon)

const _LabelSection = React.memo(
  (props: _SectionProps & { labelPlaceholder: string }) => {
    const { planParams, updatePlanParams } = useSimulation()
    const { entryId, labelPlaceholder, location } = props
    const entry = useMemo(
      () =>
        fGet(
          optGet(
            getLabeledAmountEntriesByLocation(planParams, location),
            entryId,
          ),
        ),
      [entryId, location, planParams],
    )

    const [value, setValue] = useState(entry.label ?? '')
    const handleChangeEntry = () => {
      const trimmed = value.trim()
      updatePlanParams('setLabelForLabeledAmount', {
        location,
        entryId,
        label: trimmed.length === 0 ? null : trimmed,
      })
    }
    const ref = useRef<HTMLInputElement>(null)
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
          ref={ref}
          type="text"
          className="bg-gray-200 px-2 py-1.5 rounded-lg w-full max-w-[400px] mt-2"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onBlur={handleChangeEntry}
          onKeyUp={(e) => {
            if (e.key === 'Enter') ref.current?.blur()
          }}
          placeholder={labelPlaceholder}
        />
      </_Section>
    )
  },
)
const _AmountSection = React.memo((props: _SectionProps) => {
  const { planParams, updatePlanParams } = useSimulation()
  const { entryId, location, hasMonthRange } = props
  const { increment, decrement } = smartDeltaFnForMonthlyAmountInput
  const entry = useMemo(
    () =>
      fGet(
        optGet(
          getLabeledAmountEntriesByLocation(planParams, location),
          entryId,
        ),
      ),
    [entryId, location, planParams],
  )

  const handleAmountChange = (amount: number) => {
    updatePlanParams('setAmountForLabeledAmount', { location, entryId, amount })
  }
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
        <div className="flex">
          <AmountInput
            className="w-[100px] text-input"
            prefix="$"
            value={entry.value}
            onChange={handleAmountChange}
            decimals={0}
            modalLabel={null}
          />
          <button
            className="ml-3 px-3"
            onClick={() => handleAmountChange(increment(entry.value))}
          >
            <FontAwesomeIcon icon={faPlus} />
          </button>
          <button
            className="px-3"
            onClick={() => handleAmountChange(decrement(entry.value))}
          >
            <FontAwesomeIcon icon={faMinus} />
          </button>
        </div>
        <RealOrNominalInput
          className="mt-4"
          nominal={entry.nominal}
          onChange={(nominal) =>
            updatePlanParams('setNominalForLabeledAmount', {
              location,
              entryId,
              nominal,
            })
          }
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
    const { planParams, planParamsExt, updatePlanParams } = useSimulation()
    const { clampMonthRangeToNow, months } = planParamsExt
    const { location, entryId } = props
    const entry = useMemo(
      () =>
        fGet(
          optGet(
            getValueForMonthRangeEntriesByLocation(planParams, location),
            entryId,
          ),
        ),
      [entryId, location, planParams],
    )

    const monthRangeClamped = clampMonthRangeToNow(entry.monthRange)
    const handleChange = (monthRange: MonthRange) => {
      updatePlanParams('setMonthRangeForValueForMonthRange', {
        location,
        entryId,
        monthRange,
      })
    }

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
              setValue={handleChange}
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
                onClick={() =>
                  handleChange({
                    type: 'startAndNumMonths',
                    start: months.now,
                    numMonths: 1,
                  })
                }
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
