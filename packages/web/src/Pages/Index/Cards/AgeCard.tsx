import { faExclamationCircle } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Switch } from '@headlessui/react'
import _ from 'lodash'
import React from 'react'
import { getDefaultParams } from '../../../TPAWSimulator/DefaultParams'
import { checkYearRange, TPAWParams } from '../../../TPAWSimulator/TPAWParams'
import { MAX_AGE } from '../../../TPAWSimulator/TPAWParamsValidator'
import { StateObj } from '../../../Utils/UseStateObj'
import { SliderInput } from '../../Common/Inputs/SliderInput/SliderInput'
import { ToggleSwitch } from '../../Common/Inputs/ToggleSwitch'
import { CardItem } from '../CardItem'

export const AgeCard = React.memo(
  ({params: paramsObj}: {params: StateObj<TPAWParams>}) => {
    const {value: params, set: setParams} = paramsObj
    const retired = params.age.start === params.age.retirement
    const subHeading = retired
      ? `Retired, Current: ${params.age.start}, Max: ${params.age.end}`
      : `Current: ${params.age.start}, Retirement: ${params.age.retirement}, Max: ${params.age.end}`

    const warn = [
      ...params.savings,
      ...params.withdrawals.fundedByBonds,
      ...params.withdrawals.fundedByRiskPortfolio,
    ].some(entry => checkYearRange(params, entry.yearRange) !== 'ok')

    return (
      <CardItem heading="Age" subHeading={subHeading}>
        <div
          className={`grid items-center gap-x-4`}
          style={{grid: ' auto / auto 1fr '}}
        >
          <Switch.Group>
            <Switch.Label className="">Are you retired?</Switch.Label>
            <ToggleSwitch
              className=""
              enabled={retired}
              setEnabled={retired => {
                const age = {...params.age}
                if (retired) {
                  age.retirement = age.start
                } else {
                  const defaultRetirementAge = getDefaultParams().age.retirement
                  age.retirement =
                    defaultRetirementAge <= age.start
                      ? Math.floor((age.end + age.start )/ 2)
                      : defaultRetirementAge
                }
                setParams({...params, age})
              }}
            />
          </Switch.Group>
        </div>
        <div className="pt-4">
          <p className="">
            {retired
              ? `Select your current age and the maximum age for planning.`
              : `Select your current age, the age your plan to retire, and the maximum age for planning.`}
          </p>
        </div>
        <SliderInput
          key={`${retired}`}
          className=""
          height={60}
          pointers={_.compact([
            {value: params.age.start, type: 'left'},
            retired
              ? undefined
              : {value: params.age.retirement, type: 'normal'},
            {value: params.age.end, type: 'right'},
          ])}
          onChange={values => {
            const [start, retirement, end] = retired
              ? [values[0], values[0], values[1]]
              : values
            const age = {start, retirement, end}
            setParams(params => ({...params, age}))
          }}
          formatValue={x => `${x}`}
          domain={_.range(MAX_AGE+1).map(value => ({
            value,
            tick:
              value % 10 === 0 ? 'large' : value % 2 === 0 ? 'small' : 'none',
          }))}
        />
        {warn && (
          <h2 className="text-red-500 mt-2">
            <FontAwesomeIcon
              className="mr-2 text-red-500"
              icon={faExclamationCircle}
            />{' '}
            {`One or more entries in the "Future Savings and Retirement Income" section or "Extra Spending" section has a year that is out-of-bounds or a starting year later than the ending year.`}
          </h2>
        )}
      </CardItem>
    )
  }
)
