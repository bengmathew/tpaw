import { assert, fGet, PlanParams } from '@tpaw/common'
import React from 'react'
import { PlanParamsHelperFns } from '@tpaw/common'
import { CenteredModal } from '../../../../../Common/Modal/CenteredModal'
import { useIANATimezoneName } from '../../../../PlanRootHelpers/WithNonPlanParams'
import {
  useSimulationInfo,
  useSimulationResultInfo,
} from '../../../../PlanRootHelpers/WithSimulation'
import { PlanMenuActionModalConvertDatingCommon } from './PlanMenuActionModalConvertDatingCommon'

export const PlanMenuActionModalConvertDatingLocal = React.memo(
  ({
    show,
    onHide,
    onConvert,
    skipNoUndoCopy,
  }: {
    show: boolean
    onHide: () => void
    onConvert: (planParams: PlanParams) => void
    skipNoUndoCopy: boolean
  }) => {
    const { simulationResult } = useSimulationResultInfo()
    const { ianaTimezoneName } = useIANATimezoneName()

    return (
      <CenteredModal show={show} onOutsideClickOrEscape={onHide}>
        <PlanMenuActionModalConvertDatingCommon
          onHide={onHide}
          onConvert={() =>
            onConvert(
              PlanParamsHelperFns.switchDating(
                simulationResult.planParamsNormOfResult,
                simulationResult.portfolioBalanceEstimationByDated
                  .currentBalance,
                ianaTimezoneName,
              ),
            )
          }
          isRunning={false}
          skipNoUndoCopy={skipNoUndoCopy}
        />
      </CenteredModal>
    )
  },
)
