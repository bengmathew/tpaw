import { faCircle } from '@fortawesome/pro-light-svg-icons'
import { faCircle as faCircleSolid } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Description, Label, Radio, RadioGroup } from '@headlessui/react'
import React from 'react'
import { useSimulationInfo } from '../../../PlanRootHelpers/WithSimulation'

export const PlanInputAgeWithdrawalStart = React.memo(
  ({
    className = '',
    style,
  }: {
    className?: string
    style?: React.CSSProperties
  }) => {
    const { planParamsNormInstant, updatePlanParams } = useSimulationInfo()
    const { person1, person2, simulationMonths } = planParamsNormInstant.ages
    if (!person2) return <></>
    if (person1.retirement.isRetired && person2.retirement.isRetired)
      return <></>

    return (
      <div className={`${className}`} style={style}>
        <RadioGroup<'div', 'person1' | 'person2'>
          value={simulationMonths.withdrawalStartMonth.atRetirementOf}
          onChange={(x: 'person1' | 'person2') =>
            updatePlanParams('setWithdrawalStart', x)
          }
        >
          <Label className="font-bold block text-lg">
            When should withdrawals start?
          </Label>
          <div className="mt-2">
            <Radio<'div', 'person1' | 'person2'>
              value={'person1'}
              className="flex items-center gap-x-2 py-0.5 cursor-pointer"
            >
              {({ checked }) => (
                <>
                  <FontAwesomeIcon icon={checked ? faCircleSolid : faCircle} />
                  <Description as="h2" className={`py-1`}>
                    {person1.retirement.isRetired ? 'Now' : 'Your retirement'}
                  </Description>
                </>
              )}
            </Radio>
            <Radio<'div', 'person1' | 'person2'>
              value={'person2'}
              className="flex items-center gap-x-2 py-0.5 cursor-pointer"
            >
              {({ checked }) => (
                <>
                  <FontAwesomeIcon icon={checked ? faCircleSolid : faCircle} />
                  <Description as="h2" className={`py-1`}>
                    {person2.retirement.isRetired
                      ? 'Now'
                      : `Your partner's retirement`}
                  </Description>
                </>
              )}
            </Radio>
          </div>
          <h2 className="mt-2">
            This will be considered the start of your retirement phase as a
            couple.
          </h2>
        </RadioGroup>
      </div>
    )
  },
)
