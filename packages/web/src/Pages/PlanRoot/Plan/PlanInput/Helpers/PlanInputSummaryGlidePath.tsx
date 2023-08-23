import { GlidePath } from '@tpaw/common'
import React, { useMemo } from 'react'
import { normalizeGlidePath } from '../../../../../TPAWSimulator/PlanParamsProcessed/PlanParamsProcessRisk'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'
import { PlanParamsExtended } from '../../../../../TPAWSimulator/ExtentPlanParams'
import { monthToStringForGlidePath } from '../../../../Common/Inputs/GlidePathInput'

export const PlanInputSummaryGlidePath = React.memo(
  ({
    className = '',
    glidePath,
    format,
  }: {
    className?: string
    glidePath: GlidePath
    format: (x: number) => string
  }) => {
    const { planParamsExt } = useSimulation()
    const { intermediate, starting } = useMemo(
      () => ({
        intermediate: planParamsExt.glidePathIntermediateValidated(
          glidePath.intermediate,
        ),
        starting: normalizeGlidePath(glidePath, planParamsExt)[0],
      }),
      [glidePath, planParamsExt],
    )

    return (
      <div
        className={`${className} inline-grid gap-x-10 items-center`}
        style={{ grid: 'auto/auto auto' }}
      >
        <h2>Now</h2>
        <h2 className="text-right">{format(starting)}</h2>
        <_Intermediate
          intermediate={intermediate.filter(
            (x) => x.issue !== 'before' && x.issue !== 'after',
          )}
          format={format}
        />
        <h2>At max age</h2>
        <h2 className="text-right">{format(glidePath.end.stocks)}</h2>
        <_Intermediate
          intermediate={intermediate.filter((x) => x.issue === 'after')}
          format={format}
        />
      </div>
    )
  },
)

const _Intermediate = React.memo(
  ({
    intermediate,
    format,
  }: {
    intermediate: ReturnType<
      PlanParamsExtended['glidePathIntermediateValidated']
    >
    format: (x: number) => string
  }) => {
    const { planParamsExt } = useSimulation()
    return (
      <>
        {intermediate.map((x, i) => (
          <React.Fragment key={i}>
            <h2 className={`${x.issue === 'none' ? '' : 'text-errorFG'}`}>
              {monthToStringForGlidePath(x.month, planParamsExt).full}
            </h2>
            <h2 className="text-right">{format(x.stocks)}</h2>
          </React.Fragment>
        ))}
      </>
    )
  },
)
