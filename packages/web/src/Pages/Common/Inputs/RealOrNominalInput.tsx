import { faCircle as faCircleLight } from '@fortawesome/pro-regular-svg-icons'
import { faCircle as faCircleSolid } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { RadioGroup } from '@headlessui/react'
import React, { useState } from 'react'
import { Contentful } from '../../../Utils/Contentful'
import { useSimulation } from '../../App/WithSimulation'
import { usePlanContent } from '../../Plan/Plan'
import { ModalBase } from '../Modal/ModalBase'

export const RealOrNominalInput = React.memo(
  ({
    className = '',
    nominal,
    onChange,
  }: {
    className?: string
    nominal: boolean
    onChange: (x: boolean) => void
  }) => {
    const { params } = useSimulation()
    const planContent = usePlanContent()
    const [showExplanation, setShowRealDollarsExplanation] = useState(false)

    return (
      <div className={`${className} flex flex-row items-center gap-x-2 mt-2`}>
        <RadioGroup
          value={nominal ? ('nominal' as const) : ('real' as const)}
          onChange={(type: 'nominal' | 'real') => onChange(type === 'nominal')}
          className="grid gap-y-2"
        >
          <RadioGroup.Label>This amount is</RadioGroup.Label>
          <RadioGroup.Option value="real">
            {({ checked }) => (
              <div className="flex items-start gap-x-2 cursor-pointer">
                <FontAwesomeIcon
                  className="text-sm mt-1.5"
                  icon={checked ? faCircleSolid : faCircleLight}
                />
                <h2 className="">adjusted for inflation (real dollars)</h2>
              </div>
            )}
          </RadioGroup.Option>
          <RadioGroup.Option value="nominal">
            {({ checked }) => (
              <div className="flex items-start gap-x-2  cursor-pointer">
                <FontAwesomeIcon
                  className="text-sm mt-1.5"
                  icon={checked ? faCircleSolid : faCircleLight}
                />
                <h2 className="">
                  not adjusted for inflation (nominal dollars)
                </h2>
              </div>
            )}
          </RadioGroup.Option>
          <RadioGroup.Description>
            <button
              className="underline text-sm"
              onClick={() => setShowRealDollarsExplanation(true)}
            >
              What does this mean?
            </button>
          </RadioGroup.Description>
        </RadioGroup>
        {showExplanation && (
          <ModalBase onClose={() => setShowRealDollarsExplanation(false)}>
            {(transitionOut) => (
              <div className="-mt-4">
                <Contentful.RichText
                  body={planContent.misc.realDollarExplanation[params.strategy]}
                  p={`p-base mt-4`}
                />
              </div>
            )}
          </ModalBase>
        )}
      </div>
    )
  },
)
