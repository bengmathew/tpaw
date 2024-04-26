import { PlanParams } from '@tpaw/common'
import React from 'react'
import { PlanParamsHelperFns } from '../../../../../../UseSimulator/PlanParamsHelperFns'
import { CenteredModal } from '../../../../../Common/Modal/CenteredModal'
import { useIANATimezoneName } from '../../../../PlanRootHelpers/WithNonPlanParams'
import { useSimulation } from '../../../../PlanRootHelpers/WithSimulation'
import { PlanMenuActionModalConvertDatingCommon } from './PlanMenuActionModalConvertDatingCommon'

export const PlanMenuActionModalConvertDatingLocal = React.memo(
  ({
    show,
    onHide,
    onConvert,
    skipNoUndoCopy
  }: {
    show: boolean
    onHide: () => void
    onConvert: (planParams: PlanParams) => void
    skipNoUndoCopy:boolean
  }) => {
    const { ianaTimezoneName } = useIANATimezoneName()
    const { planParamsNorm, currentPortfolioBalanceInfo } = useSimulation()
    return (
      <CenteredModal show={show} onOutsideClickOrEscape={onHide}>
        <PlanMenuActionModalConvertDatingCommon
          onHide={onHide}
          onConvert={() =>
            onConvert(
              PlanParamsHelperFns.switchDating(
                planParamsNorm,
                currentPortfolioBalanceInfo,
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
