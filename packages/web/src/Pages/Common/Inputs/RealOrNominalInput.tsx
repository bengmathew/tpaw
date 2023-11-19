import { faCircle as faCircleLight } from '@fortawesome/pro-regular-svg-icons'
import { faCircle as faCircleSolid } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { RadioGroup } from '@headlessui/react'
import React, { useState } from 'react'
import { Contentful } from '../../../Utils/Contentful'
import { usePlanContent } from '../../PlanRoot/PlanRootHelpers/WithPlanContent'
import { useSimulation } from '../../PlanRoot/PlanRootHelpers/WithSimulation'
import { CenteredModal } from '../Modal/CenteredModal'

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
    const { planParams } = useSimulation()
    const planContent = usePlanContent()
    const [showExplanation, setShowExplanation] = useState(false)

    return (
      <div className={`${className} flex flex-row items-center gap-x-2`}>
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
              onClick={() => setShowExplanation(true)}
            >
              What does this mean?
            </button>
          </RadioGroup.Description>
        </RadioGroup>
        <CenteredModal
          className=" dialog-outer-div"
          show={showExplanation}
          onOutsideClickOrEscape={() => setShowExplanation(false)}
        >
          <h2 className=" dialog-heading">{`What does "adjusted for inflation" mean?`}</h2>
          <div className=" dialog-content-div">
            <Contentful.RichText
              body={
                planContent.misc.realDollarExplanation[
                  planParams.advanced.strategy
                ]
              }
              p={`p-base mt-3`}
            />
          </div>
        </CenteredModal>
      </div>
    )
  },
)
