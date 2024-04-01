import {
  faCaretDown,
  faCaretRight,
  faCircle,
  faTrash,
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Switch } from '@headlessui/react'
import {
  CalendarMonthFns,
  LabeledAmountTimedLocation,
  block,
  fGet,
  noCase,
} from '@tpaw/common'
import { clsx } from 'clsx'
import _ from 'lodash'
import React, {
  ReactNode,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { NormalizedLabeledAmountTimed } from '../../../../../UseSimulator/NormalizePlanParams/NormalizeLabeledAmountTimedList/NormalizeLabeledAmountTimedList'
import { InMonthsFns } from '../../../../../Utils/InMonthsFns'
import { yourOrYourPartners } from '../../../../../Utils/YourOrYourPartners'
import { CheckBox } from '../../../../Common/Inputs/CheckBox'
import { CalendarMonthInput } from '../../../../Common/Inputs/MonthInput/CalendarMonthInput'
import { InMonthsInput } from '../../../../Common/Inputs/MonthInput/InMonthsInput'
import { CenteredModal } from '../../../../Common/Modal/CenteredModal'
import { LabeledAmountTimedDisplay } from '../../../../Common/LabeledAmountTimedDisplay'
import {
  RemovePartnerAdjustments,
  getRemovePartnerAdjustments,
} from '../../../PlanRootHelpers/GetPlanParamsChangeActionImpl/GetDeletePartnerChangeActionImpl'
import {
  RetirePersonAdjustments,
  getRetirePersonAdjustments,
} from '../../../PlanRootHelpers/GetPlanParamsChangeActionImpl/GetSetPersonRetiredChangeActionImpl'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'
import { PlanInputSummaryGlidePath } from '../Helpers/PlanInputSummaryGlidePath'
import { planSectionLabel } from '../Helpers/PlanSectionLabel'
import { PlanInputAgeOpenableSection } from './PlanInputAge'

export const PlanInputAgePerson = React.memo(
  ({
    personType,
    className = '',
    style,
    openSection,
    setOpenSection,
  }: {
    personType: 'person1' | 'person2'
    className?: string
    style?: React.CSSProperties
    openSection: PlanInputAgeOpenableSection
    setOpenSection: (x: PlanInputAgeOpenableSection) => void
  }) => {
    const { planParamsNorm, updatePlanParams } = useSimulation()
    const divRef = useRef<HTMLDivElement>(null)
    const person = fGet(planParamsNorm.ages[personType])

    const retireAdjustments = useMemo(
      () => getRetirePersonAdjustments(personType, planParamsNorm),
      [personType, planParamsNorm],
    )

    const [showRetireWarnings, setShowRetireWarnings] = useState(false)

    const deletePartnerAdjustments = useMemo(
      () => getRemovePartnerAdjustments(planParamsNorm),
      [planParamsNorm],
    )
    const [showDeletePartnerWarnings, setShowDeletePartnerWarnings] =
      useState(false)

    useState<ReactNode[]>([])
    return (
      <div
        className={`${className} `}
        style={style}
        ref={divRef}
        onClick={(e) => {
          if (e.target === divRef.current) setOpenSection(`none`)
        }}
      >
        <div
          className="flex justify-between"
          onClick={() => setOpenSection('none')}
        >
          <h2 className="font-bold text-lg">
            {personType === 'person1' ? 'You' : 'Your Partner'}
          </h2>
          {personType === 'person2' && (
            <button
              className=""
              onClick={() => {
                if (deletePartnerAdjustments) {
                  setShowDeletePartnerWarnings(true)
                } else {
                  updatePlanParams('deletePartner', null)
                }
              }}
            >
              <FontAwesomeIcon className="mr-2" icon={faTrash} />
            </button>
          )}
        </div>
        <div
          className="flex items-center col-span-2 gap-x-4 mt-4 mb-2"
          onClick={() => setOpenSection('none')}
        >
          <Switch.Group>
            <Switch.Label className="">
              {personType === 'person1'
                ? 'Are you retired?'
                : 'Is your partner retired?'}
            </Switch.Label>
            <CheckBox
              className=""
              enabled={person.retirement.isRetired}
              setEnabled={(retired) => {
                if (retired) {
                  if (retireAdjustments) {
                    setShowRetireWarnings(true)
                  } else {
                    updatePlanParams('setPersonRetired', personType)
                  }
                } else {
                  updatePlanParams('setPersonNotRetired', personType)
                }
              }}
            />
          </Switch.Group>
        </div>

        <div
          className="grid items-center "
          style={{ grid: 'auto / 145px 1fr' }}
        >
          <_MonthOfBirthInput
            sectionName="Month of Birth"
            personType={personType}
            sectionType={`${personType}-monthOfBirth`}
            currSection={openSection}
            setCurrSection={setOpenSection}
          />
          {!person.retirement.isRetired && (
            <_AgeInput
              sectionName="Retirement Age"
              personType={personType}
              type="retirementAge"
              sectionType={`${personType}-retirementAge`}
              currSection={openSection}
              setCurrSection={setOpenSection}
            />
          )}
          <_AgeInput
            sectionName="Max Age"
            personType={personType}
            type="maxAge"
            sectionType={`${personType}-maxAge`}
            currSection={openSection}
            setCurrSection={setOpenSection}
          />
        </div>
        <div
          className="grid gap-y-2 items-center gap-x-4"
          style={{ grid: 'auto / 145px 1fr' }}
        ></div>

        <DeletePartnerWarningsModal
          show={showDeletePartnerWarnings}
          onCancel={() => {
            setShowDeletePartnerWarnings(false)
          }}
          onApply={() => {
            setShowDeletePartnerWarnings(false)
            updatePlanParams('deletePartner', null)
          }}
          adjustments={deletePartnerAdjustments}
        />
        <RetirementWarningsModal
          personType={personType}
          show={showRetireWarnings}
          onCancel={() => {
            setShowRetireWarnings(false)
          }}
          onApply={() => {
            setShowRetireWarnings(false)
            updatePlanParams('setPersonRetired', personType)
          }}
          adjustments={retireAdjustments}
        />
      </div>
    )
  },
)

export const _Section = React.memo(
  ({
    sectionName,
    sectionType,
    currSection,
    setCurrSection,
    children: [summaryChild, editChild],
  }: {
    sectionName: string
    sectionType: Exclude<PlanInputAgeOpenableSection, 'none'>
    currSection: PlanInputAgeOpenableSection
    setCurrSection: (open: PlanInputAgeOpenableSection) => void
    children: [React.ReactNode, React.ReactNode]
  }) => {
    const buttonDivRef = useRef<HTMLDivElement>(null)

    return currSection !== sectionType ? (
      <>
        <button
          className="text-start py-1.5 self-start whitespace-nowrap"
          onClick={() => setCurrSection(sectionType)}
        >
          <FontAwesomeIcon className="text-xs mr-2" icon={faCaretRight} />
          {sectionName}
        </button>
        <div
          className=""
          ref={buttonDivRef}
          onClick={(e) => {
            if (e.target === buttonDivRef.current) setCurrSection('none')
          }}
        >
          <button
            className="text-start pl-2 py-1.5"
            onClick={() => setCurrSection(sectionType)}
          >
            {summaryChild}
          </button>
        </div>
      </>
    ) : (
      <div className={`col-span-2 bg-gray-100 rounded-xl p-2 my-2`}>
        <button
          className="py-1.5 text-start "
          onClick={() => setCurrSection('none')}
        >
          <FontAwesomeIcon className="text-xs mr-2" icon={faCaretDown} />
          {sectionName}
        </button>
        {editChild}
      </div>
    )
  },
)

export const _MonthOfBirthInput = React.memo(
  ({
    sectionName,
    personType,
    sectionType,
    currSection,
    setCurrSection,
  }: {
    sectionName: string
    personType: 'person1' | 'person2'
    sectionType: Exclude<PlanInputAgeOpenableSection, 'none'>
    currSection: PlanInputAgeOpenableSection
    setCurrSection: (open: PlanInputAgeOpenableSection) => void
  }) => {
    const { planParamsNorm, updatePlanParams } = useSimulation()
    const person = fGet(planParamsNorm.ages[personType])
    const { monthOfBirth, currentAge } = person

    return (
      <_Section
        sectionName={sectionName}
        sectionType={sectionType}
        currSection={currSection}
        setCurrSection={setCurrSection}
      >
        {CalendarMonthFns.toStr(monthOfBirth.baseValue)}
        <div className="">
          <CalendarMonthInput
            className="mt-2 ml-4"
            normValue={{ ...monthOfBirth }}
            onChange={(monthOfBirth) =>
              updatePlanParams('setPersonMonthOfBirth2', {
                person: personType,
                monthOfBirth,
              })
            }
          />
          <h2 className="mb-2 mt-3 ml-4">
            Age: {InMonthsFns.toStr(currentAge)}
          </h2>
        </div>
      </_Section>
    )
  },
)
export const _AgeInput = React.memo(
  ({
    sectionName,
    personType,
    type,
    sectionType,
    currSection,
    setCurrSection,
  }: {
    sectionName: string
    personType: 'person1' | 'person2'
    type: 'retirementAge' | 'maxAge'
    sectionType: Exclude<PlanInputAgeOpenableSection, 'none'>
    currSection: PlanInputAgeOpenableSection
    setCurrSection: (open: PlanInputAgeOpenableSection) => void
  }) => {
    const { planParamsNorm, updatePlanParams } = useSimulation()
    const age = (() => {
      const person = fGet(planParamsNorm.ages[personType])
      if (type === 'retirementAge') return fGet(person.retirement.ageIfInFuture)
      if (type === 'maxAge') return person.maxAge
      noCase(type)
    })()

    return (
      <_Section
        sectionName={sectionName}
        sectionType={sectionType}
        currSection={currSection}
        setCurrSection={setCurrSection}
      >
        {InMonthsFns.toStr(age.baseValue)}
        <InMonthsInput
          className="mt-2 mb-4 ml-4"
          modalLabel={sectionName}
          normValue={{
            baseValue: age.baseValue,
            validRangeInMonths: {
              includingLocalConstraints: age.validRangeInMonths,
            },
          }}
          onChange={(inMonths) =>
            type === 'retirementAge'
              ? updatePlanParams('setPersonRetirementAge', {
                  person: personType,
                  retirementAge: inMonths,
                })
              : type === 'maxAge'
                ? updatePlanParams('setPersonMaxAge', {
                    person: personType,
                    maxAge: inMonths,
                  })
                : noCase(type)
          }
        />
      </_Section>
    )
  },
)

const RetirementWarningsModal = React.memo(
  ({
    show,
    onCancel,
    onApply,
    adjustments: adjustmentsIn,
    personType,
  }: {
    show: boolean
    onCancel: () => void
    onApply: () => void
    adjustments: RetirePersonAdjustments | null
    personType: 'person1' | 'person2'
  }) => {
    const { planParamsNorm } = useSimulation()
    const [adjustments, setAdjustments] = useState(
      null as null | RetirePersonAdjustments,
    )
    const handleShow = (show: boolean) => {
      if (show) setAdjustments(fGet(adjustmentsIn))
    }
    const handleShowRef = useRef(handleShow)
    handleShowRef.current = handleShow
    useLayoutEffect(() => {
      handleShowRef.current(show)
    }, [show])

    return (
      <CenteredModal
        className=" dialog-outer-div"
        show={show}
        onOutsideClickOrEscape={null}
      >
        {adjustments &&
          block(() => {
            const {
              futureSavingsEntriesToBeRemovedDueSectionRemoval,
              valueForMonthRangeEntriesToBeAdjusted,
              spawAndSWRStockAllocationAdjusted,
            } = adjustments
            return (
              <>
                <h2 className=" text-2xl font-bold">Additional Changes</h2>
                <div className=" dialog-content-div">
                  <p className="p-base">
                    Setting {yourOrYourPartners(personType)} status to retired
                    will cause the following additional changes to be made to
                    the plan:
                  </p>
                  <_FutureSavingsIssue
                    entries={futureSavingsEntriesToBeRemovedDueSectionRemoval}
                  />
                  <_MonthRangesBySectionIssue
                    description={`The following entries reference ${yourOrYourPartners(personType)} retirement and or last working month. They will be adjusted to "now" or "in the past" respectively.`}
                    valueForMonthRanges={valueForMonthRangeEntriesToBeAdjusted}
                  />
                  {spawAndSWRStockAllocationAdjusted && (
                    <_StockAllocationIssueSection
                      referenceStr={`${yourOrYourPartners(personType)} retirement`}
                    />
                  )}
                </div>
                <div className=" dialog-button-div">
                  <button className=" dialog-button-cancel" onClick={onCancel}>
                    Cancel
                  </button>
                  <button className=" dialog-button-dark" onClick={onApply}>
                    Apply Changes
                  </button>
                </div>
              </>
            )
          })}
      </CenteredModal>
    )
  },
)

const DeletePartnerWarningsModal = React.memo(
  ({
    show,
    onCancel,
    onApply,
    adjustments: adjustmentsIn,
  }: {
    show: boolean
    onCancel: () => void
    onApply: () => void
    adjustments: RemovePartnerAdjustments | null
  }) => {
    const [adjustments, setAdjustments] = useState(
      null as null | RemovePartnerAdjustments,
    )
    const handleShow = (show: boolean) => {
      if (show) setAdjustments(fGet(adjustmentsIn))
    }
    const handleShowRef = useRef(handleShow)
    handleShowRef.current = handleShow
    useLayoutEffect(() => {
      handleShowRef.current(show)
    }, [show])
    return (
      <CenteredModal
        className=" dialog-outer-div"
        show={show}
        onOutsideClickOrEscape={null}
      >
        {adjustments &&
          block(() => {
            const {
              futureSavingsEntriesToBeRemovedDueSectionRemoval,
              amountForMonthRangeEntriesToBeRemoved,
              spawAndSWRStockAllocationAdjusted,
            } = adjustments
            return (
              <>
                <h2 className=" font-bold text-2xl">Additional Changes</h2>
                <div className=" dialog-content-div">
                  <p className="p-base">
                    Deleting your partner will cause the following additional
                    changes to be made to the plan:
                  </p>
                  <_FutureSavingsIssue
                    entries={futureSavingsEntriesToBeRemovedDueSectionRemoval}
                  />
                  <_MonthRangesBySectionIssue
                    description="The following entries reference your partner. They will be removed."
                    valueForMonthRanges={amountForMonthRangeEntriesToBeRemoved}
                  />
                  {spawAndSWRStockAllocationAdjusted && (
                    <_StockAllocationIssueSection
                      referenceStr={`your partner`}
                    />
                  )}
                </div>
                <div className=" dialog-button-div">
                  <button className=" dialog-button-cancel" onClick={onCancel}>
                    Cancel
                  </button>
                  <button className=" dialog-button-dark" onClick={onApply}>
                    Apply Changes
                  </button>
                </div>
              </>
            )
          })}
      </CenteredModal>
    )
  },
)

const _FutureSavingsIssue = React.memo(
  ({ entries }: { entries: NormalizedLabeledAmountTimed[] }) => {
    if (entries.length === 0) return <></>
    return (
      <div className={'mt-6'}>
        <h2 className="font-bold text-xl mt-6">Future Savings</h2>
        <p className="p-base mt-2">
          {`The future savings section will no longer be applicable. It has the following entries that will be removed:`}
        </p>
        <_MonthRangesIssueList entries={entries} />
      </div>
    )
  },
)

const _MonthRangesBySectionIssue = React.memo(
  ({
    valueForMonthRanges,
    description,
  }: {
    description: string
    valueForMonthRanges: Map<
      LabeledAmountTimedLocation,
      NormalizedLabeledAmountTimed[]
    >
  }) => {
    if (valueForMonthRanges.size === 0) return <></>
    return (
      <div className={'mt-4'}>
        <h2 className="font-bold text-xl mt-6">Month Ranges</h2>
        <p className="p-base mt-2">{description}</p>
        {_.sortBy([...valueForMonthRanges.entries()], ([location]) => {
          switch (location) {
            case 'futureSavings':
              return 0
            case 'incomeDuringRetirement':
              return 1
            case 'extraSpendingEssential':
              return 2
            case 'extraSpendingDiscretionary':
              return 3
            default:
              noCase(location)
          }
        }).map(([location, entries]) => (
          <div key={location} className="mt-4">
            <h2 className="font-bold">
              {block(() => {
                switch (location) {
                  case 'futureSavings':
                    return planSectionLabel('future-savings')
                  case 'incomeDuringRetirement':
                    return planSectionLabel('income-during-retirement')
                  case 'extraSpendingEssential':
                    return `${planSectionLabel('extra-spending')} - Essential`
                  case 'extraSpendingDiscretionary':
                    return `${planSectionLabel('extra-spending')} - Discretionary`
                  default:
                    noCase(location)
                }
              })}
            </h2>
            <_MonthRangesIssueList className="" entries={entries} />
          </div>
        ))}
      </div>
    )
  },
)

const _MonthRangesIssueList = React.memo(
  ({
    className,
    entries,
  }: {
    className?: string
    entries: NormalizedLabeledAmountTimed[]
  }) => {
    return (
      <div className={clsx(className)}>
        {entries.map((entry) => (
          <div className="mt-1 flex gap-x-3" key={entry.id}>
            <FontAwesomeIcon className="text-[6px] mt-2.5" icon={faCircle} />
            <LabeledAmountTimedDisplay
              className=""
              labelClassName="font-medium"
              entry={entry}
            />
          </div>
        ))}
      </div>
    )
  },
)

const _StockAllocationIssueSection = React.memo(
  ({ referenceStr }: { referenceStr: string }) => {
    const { planParamsNorm } = useSimulation()
    return (
      <div className={clsx('mt-6')}>
        <h2 className="font-bold text-xl mt-6">Stock Allocation</h2>
        <p className="p-base mt-2">
          {`One or more entries in the stock allocation table in the ${planSectionLabel('risk')} section reference 
                        ${referenceStr}. The
                         corresponding entries will be removed from the table.`}
        </p>
        {planParamsNorm.advanced.strategy === 'TPAW' && (
          <p className="p-base mt-2">
            {`Note: The stock allocation table is visible only when the strategy
                          is set to "SPAW" or "SWR".`}
          </p>
        )}
        <PlanInputSummaryGlidePath
          className="mt-2"
          normValue={planParamsNorm.risk.spawAndSWR.allocation}
        />
      </div>
    )
  },
)
