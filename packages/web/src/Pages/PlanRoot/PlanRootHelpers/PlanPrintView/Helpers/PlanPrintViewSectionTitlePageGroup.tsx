import React from 'react'
import { PlanPrintViewPageGroup } from './PlanPrintViewPageGroup'
import { PlanPrintViewArgs } from '../PlanPrintViewArgs'

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
