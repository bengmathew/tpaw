import { faCircle } from '@fortawesome/pro-light-svg-icons'
import { faCircle as faCircleSolid } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { RadioGroup } from '@headlessui/react'
import _ from 'lodash'
import React from 'react'
import { assert } from '../../../../Utils/Utils'
import { useSimulation } from '../../../App/WithSimulation'

export const PlanInputAgeWithdrawalStart = React.memo(
  ({
    className = '',
    style,
  }: {
    className?: string
    style?: React.CSSProperties
  }) => {
    const { params, setPlanParams, paramsExt } = useSimulation()
    const { people } = params.plan
    const { isPersonRetired } = paramsExt
    if (!people.withPartner) return <></>
    if (isPersonRetired('person1') && isPersonRetired('person2')) return <></>

    return (
      <div className={`${className}`} style={style}>
        <RadioGroup<'div', 'person1' | 'person2'>
          value={people.withdrawalStart}
          onChange={(x: 'person1' | 'person2') =>
            setPlanParams((plan) => {
              const clone = _.cloneDeep(plan)
              assert(clone.people.withPartner)
              clone.people.withdrawalStart = x
              return clone
            })
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
                    {isPersonRetired('person1') ? 'Now' : 'Your retirement'}
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
                    {isPersonRetired('person2')
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
