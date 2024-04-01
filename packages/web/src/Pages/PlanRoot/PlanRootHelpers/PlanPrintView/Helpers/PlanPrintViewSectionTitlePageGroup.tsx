import React from 'react'
import { PlanPrintViewArgs } from '../PlanPrintViewArgs'
import { PlanPrintViewPageGroup } from './PlanPrintViewPageGroup'

export const PlanPrintViewSectionTitlePageGroup = React.memo(
  ({
    title,
    settings,
  }: {
    title: string
    settings: PlanPrintViewArgs['settings']
  }) => {
    return (
      <PlanPrintViewPageGroup
        className="grid items-center"
        style={{ grid: 'auto/auto' }}
        settings={settings}
      >
        <h2 className="text-center font-bold text-4xl">{title}</h2>
      </PlanPrintViewPageGroup>
    )
  },
)
