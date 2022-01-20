import { faExclamationCircle } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Switch } from '@headlessui/react'
import _ from 'lodash'
import React from 'react'
import { getDefaultParams } from '../../../TPAWSimulator/DefaultParams'
import { MAX_AGE } from '../../../TPAWSimulator/TPAWParams'
import { Contentful } from '../../../Utils/Contentful'
import { joinWithCommaAnd } from '../../../Utils/JoinWithAnd'
import { useSimulation } from '../../App/WithSimulation'
import { SliderInput } from '../../Common/Inputs/SliderInput/SliderInput'
import { ToggleSwitch } from '../../Common/Inputs/ToggleSwitch'
import { usePlanContent } from '../Plan'
import { paramsInputValidate } from './Helpers/ParamInputValidate'
import { paramsInputLabel } from './Helpers/ParamsInputLabel'

export const ParamsInputAge = React.memo(() => {
  const content = usePlanContent()
  const {params, setParams} = useSimulation()
  const retired = params.age.start === params.age.retirement
  const subHeading = retired
    ? `Retired, Current: ${params.age.start}, Max: ${params.age.end}`
    : `Current: ${params.age.start}, Retirement: ${params.age.retirement}, Max: ${params.age.end}`

  const warnings = _.compact([
    !paramsInputValidate(params, 'futureSavings')
      ? paramsInputLabel('future-savings')
      : undefined,
    !paramsInputValidate(params, 'retirementIncome')
      ? paramsInputLabel('income-during-retirement')
      : undefined,
    !paramsInputValidate(params, 'extraSpending')
      ? paramsInputLabel('extra-spending')
      : undefined,
  ]).map(x => `"${x}"`)

  return (
    <div className="">
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
                    ? Math.floor((age.end + age.start) / 2)
                    : defaultRetirementAge
              }
              setParams({...params, age})
            }}
          />
        </Switch.Group>
      </div>
      <div className="pt-4">
        <Contentful.RichText
          body={
            retired
              ? content.age.introRetired.fields.body
              : content.age.introNotRetired.fields.body
          }
          p=""
        />
      </div>
      <SliderInput
        key={`${retired}`}
        className="max-w-[500px]"
        height={60}
        pointers={_.compact([
          {value: params.age.start, type: 'left'},
          retired ? undefined : {value: params.age.retirement, type: 'normal'},
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
        domain={_.range(MAX_AGE + 1).map(value => ({
          value,
          tick: value % 10 === 0 ? 'large' : value % 2 === 0 ? 'small' : 'none',
        }))}
      />
      {warnings.length > 0 && (
        <h2 className="text-red-500 mt-2">
          <FontAwesomeIcon
            className="mr-2 text-red-500"
            icon={faExclamationCircle}
          />{' '}
          {`One or more entries in the ${joinWithCommaAnd(warnings)} ${
            warnings.length > 1 ? 'sections' : 'section'
          } has a year that is out-of-bounds or a starting year later than the ending year.`}
        </h2>
      )}
    </div>
  )
})
