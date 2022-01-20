import {faExclamation} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import _ from 'lodash'
import {default as React} from 'react'
import {Contentful} from '../../../Utils/Contentful'
import {useTPAWParams} from '../../App/UseTPAWParams'
import {useSimulation} from '../../App/WithSimulation'
import {usePlanContent} from '../Plan'
import {
  ByYearSchedule,
  ByYearScheduleEntry,
} from './ByYearSchedule/ByYearSchedule'
import {paramsInputValidateYearRange} from './Helpers/ParamInputValidate'

export const ParamsInputFutureSavings = React.memo(
  ({onBack}: {onBack: () => void}) => {
    const {age} = useTPAWParams().params
    const content = usePlanContent()
    if (age.start === age.retirement) return <_Retired onBack={onBack} />
    return (
      <div className="">
        <Contentful.RichText
          body={content.futureSavings.intro.fields.body}
          p=""
        />
        <ByYearSchedule
          className=""
          type="beforeRetirement"
          heading={null}
          addHeading="Add to Savings"
          editHeading="Edit Savings Entry"
          defaultYearRange={{start: 'start', end: 'lastWorkingYear'}}
          entries={params => params.savings}
          validateYearRange={(params, x) =>
            paramsInputValidateYearRange(params, 'futureSavings', x)
          }
        />
      </div>
    )
  }
)

const _Retired = React.memo(({onBack}: {onBack: () => void}) => {
  const {params, setParams} = useSimulation()
  return (
    <div>
      <p className="text-errorFG">
        <span className="px-2 py-0.5 mr-2 text-[11px] rounded-full bg-errorBlockBG text-errorBlockFG">
          <span className="text-sm">Warning</span>{' '}
          <FontAwesomeIcon icon={faExclamation} />
        </span>
        {`You are currently retired but still have the following entries for
        future savings. These will be ignored. Further contributions towards
        your retirement should be entered in the "Retirement Income" section.`}
      </p>
      <button
        className="btn-sm btn-dark mt-4"
        onClick={() => {
          setParams(params => {
            const p = _.cloneDeep(params)
            p.savings = []
            return p
          })
          onBack()
        }}
      >
        Clear Entries
      </button>
      <div className={` flex flex-col gap-y-6 mt-4 `}>
        {params.savings.map((entry, i) => (
          <ByYearScheduleEntry
            key={i}
            className=""
            params={params}
            validation={'ok'}
            entry={entry}
            onEdit={null}
            onChangeAmount={null}
          />
        ))}
      </div>
    </div>
  )
})
