import { faCircle as faCircleLight } from '@fortawesome/pro-regular-svg-icons'
import { faCircle as faCircleSolid } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { RadioGroup } from '@headlessui/react'
import React, { useState } from 'react'
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
            <p className="p-base">
              {`If the amount that you are entering is inflation adjusted, then you are entering the value in today's dollars. For example, if thirty years from now $10 will buy only what $5 buys today, then you would enter $10 thirty years from now as just $5. So a dollar entered for any age has the same buying power as a dollar does today.`}
            </p>
            <p className="p-base mt-3">{`Dollars that are adjusted for inflation in this way are called “real” dollars. Dollars that are not adjusted for inflation are called “nominal dollars.”`}</p>
          </div>
        </CenteredModal>
      </div>
    )
  },
)
