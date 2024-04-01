import { faCircle } from '@fortawesome/pro-light-svg-icons'
import { faCircle as faCircleSolid } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { RadioGroup } from '@headlessui/react'
import React from 'react'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'

export const PlanInputAgeWithdrawalStart = React.memo(
  ({
    className = '',
    style,
  }: {
    className?: string
    style?: React.CSSProperties
  }) => {
    const { planParamsNorm, updatePlanParams } = useSimulation()
    const { person1, person2, simulationMonths } = planParamsNorm.ages
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
          <RadioGroup.Label className="font-bold block text-lg">
            When should withdrawals start?
          </RadioGroup.Label>
          <div className="mt-2">
            <RadioGroup.Option<'div', 'person1' | 'person2'>
              value={'person1'}
              className="flex items-center gap-x-2 py-0.5 cursor-pointer"
            >
              {({ checked }) => (
                <>
                  <FontAwesomeIcon icon={checked ? faCircleSolid : faCircle} />
                  <RadioGroup.Description as="h2" className={`py-1`}>
                    {person1.retirement.isRetired ? 'Now' : 'Your retirement'}
                  </RadioGroup.Description>
                </>
              )}
            </RadioGroup.Option>
            <RadioGroup.Option<'div', 'person1' | 'person2'>
              value={'person2'}
              className="flex items-center gap-x-2 py-0.5 cursor-pointer"
            >
              {({ checked }) => (
                <>
                  <FontAwesomeIcon icon={checked ? faCircleSolid : faCircle} />
                  <RadioGroup.Description as="h2" className={`py-1`}>
                    {person2.retirement.isRetired
                      ? 'Now'
                      : `Your partner's retirement`}
                  </RadioGroup.Description>
                </>
              )}
            </RadioGroup.Option>
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
