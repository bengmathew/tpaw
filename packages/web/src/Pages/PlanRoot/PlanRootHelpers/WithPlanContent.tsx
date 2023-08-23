import React, { ReactElement } from 'react'
import { createContext } from '../../../Utils/CreateContext'
import { PlanContent } from './PlanRootGetStaticProps'

const [Context, usePlanContent] = createContext<PlanContent>('PlanContent')
export { usePlanContent }

export const WithPlanContent = React.memo(
  ({
    children,
    planContent,
  }: {
    children: ReactElement
    planContent: PlanContent
  }) => {
    return <Context.Provider value={planContent}>{children}</Context.Provider>
  },
)
