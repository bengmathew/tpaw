import { defaultNonPlanParams } from '@tpaw/common'
import _ from 'lodash'
import Link from 'next/link'
import React from 'react'
import { paddingCSS } from '../../../../../Utils/Geometry'
import { ToggleSwitch } from '../../../../Common/Inputs/ToggleSwitch'
import { useNonPlanParams } from '../../../PlanRootHelpers/WithNonPlanParams'
import { useGetPlanResultsChartURL } from '../../PlanResults/UseGetPlanResultsChartURL'
import { PlanInputModifiedBadge } from '../Helpers/PlanInputModifiedBadge'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from '../PlanInputBody/PlanInputBody'

export const PlanInputDevMisc = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    return (
      <PlanInputBody {...props}>
        <div className="">
          <_MiscCard className="mt-10" props={props} />
        </div>
      </PlanInputBody>
    )
  },
)

const _MiscCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { nonPlanParams, setNonPlanParams } = useNonPlanParams()
    const getPlanChartURL = useGetPlanResultsChartURL()

    const isModified = useIsPlanInputDevMiscModified()

    const handleChangeShowAllMonths = (x: boolean) => {
      const clone = _.cloneDeep(nonPlanParams)
      clone.dev.alwaysShowAllMonths = x
      setNonPlanParams(clone)
    }

    return (
      <div
        className={`${className} params-card relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
        <div className=" flex justify-start gap-x-4 items-center mt-4">
          <h2 className=""> Show All Months</h2>
          <ToggleSwitch
            className=""
            checked={nonPlanParams.dev.alwaysShowAllMonths}
            setChecked={(x) => handleChangeShowAllMonths(x)}
          />
        </div>

        <Link
          className="block underline pt-4"
          href={getPlanChartURL('asset-allocation-total-portfolio')}
          shallow
        >
          Show Asset Allocation of Total Portfolio Graph
        </Link>

        <button className="block btn-sm btn-outline mt-4" onClick={() => {}}>
          Test
        </button>
        <button
          className="block btn-sm btn-outline mt-4"
          onClick={() => {
            throw new Error('Crash Test')
          }}
        >
          Crash
        </button>
        <button
          className="mt-6 underline disabled:lighten-2 block"
          onClick={() =>
            handleChangeShowAllMonths(
              defaultNonPlanParams.dev.alwaysShowAllMonths,
            )
          }
          disabled={!isModified}
        >
          Reset to Default
        </button>
      </div>
    )
  },
)

export const useIsPlanInputDevMiscModified = () => {
  const { nonPlanParams } = useNonPlanParams()
  return (
    nonPlanParams.dev.alwaysShowAllMonths !==
    defaultNonPlanParams.dev.alwaysShowAllMonths
  )
}

export const PlanInputDevMiscSummary = React.memo(() => {
  const { nonPlanParams } = useNonPlanParams()
  return (
    <>
      <h2>
        Always show all months:{' '}
        {nonPlanParams.dev.alwaysShowAllMonths ? 'true' : 'false'}
      </h2>
    </>
  )
})
